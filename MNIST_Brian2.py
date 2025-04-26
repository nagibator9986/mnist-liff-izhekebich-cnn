import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gc
import psutil
from brian2 import *
import pickle
import gzip

def get_labeled_data(path, bTrain=True):
    """Загружает данные MNIST."""
    file_name = os.path.join(path, 'training.pkl.gz' if bTrain else 'testing.pkl.gz')
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"MNIST file {file_name} not found")
    with gzip.open(file_name, 'rb') as f:
        data = pickle.load(f)
    return {'X': data[0].reshape((-1, 784)).astype(np.float32), 'y': data[1].astype(np.int32)}

def get_matrix_from_file(file_name, shape):
    """Загружает веса из файла .npy."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Weight file {file_name} not found")
    weight_list = np.load(file_name, allow_pickle=True)
    weight_matrix = np.zeros(shape, dtype=np.float32)
    for i, j, w in weight_list:
        weight_matrix[int(i), int(j)] = w
    return weight_matrix

def normalize_weights(connections, weight_sum, conn_name='XeAe'):
    """Нормализует веса соединений."""
    weights = connections[conn_name].w
    col_sums = np.sum(weights, axis=0)
    col_sums[col_sums == 0] = 1
    weights *= weight_sum / col_sums

def get_new_assignments(result_monitor, input_numbers, n_e):
    """Назначает классы нейронам."""
    assignments = np.ones(n_e, dtype=np.int32) * -1
    max_rates = np.zeros(n_e, dtype=np.float32)
    input_nums = np.asarray(input_numbers, dtype=np.int32)
    for j in range(10):
        mask = input_nums == j
        num_inputs = np.sum(mask)
        if num_inputs > 0:
            rate = np.sum(result_monitor[mask], axis=0) / num_inputs
            for i in range(n_e):
                if rate[i] > max_rates[i]:
                    max_rates[i] = rate[i]
                    assignments[i] = j
    return assignments

def get_recognized_number_ranking(assignments, spike_rates):
    """Предсказывает классы."""
    summed_rates = np.zeros(10, dtype=np.float32)
    num_assignments = np.zeros(10, dtype=np.float32)
    for i in range(10):
        mask = assignments == i
        num_assignments[i] = np.sum(mask)
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[mask]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def plot_2d_input_weights(weights, n_input, n_e, save_path, layer_name='layer1'):
    """Визуализирует веса как тепловую карту."""
    weights_2d = weights.reshape((28, 28, n_e))
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < min(n_e, 100):
            ax.imshow(weights_2d[:, :, i], cmap='hot_r', interpolation='nearest')
            ax.axis('off')
    plt.savefig(os.path.join(save_path, f'weights_{layer_name}.png'))
    plt.close()

def main(args):
    # Настройка многопоточности через переменную окружения
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    # Настройка Brian2
    set_device('cpp_standalone', directory=None)
    prefs.codegen.target = 'cython'
    defaultclock.dt = 0.5 * ms

    # Параметры
    n_input = 784
    n_e = args.n_e
    n_i = n_e
    n_e2 = args.n_e2
    single_example_time = args.example_time * second
    resting_time = args.resting_time * second
    num_examples = args.num_examples
    update_interval = args.update_interval
    weight_update_interval = args.weight_update_interval
    save_interval = args.save_interval
    data_path = args.data_path
    save_path = args.save_path
    test_mode = args.test_mode
    input_intensity = args.input_intensity
    weight_sum = args.weight_sum

    # Создание директорий
    for subdir in ['', 'weights', 'activity', 'checkpoints']:
        os.makedirs(os.path.join(save_path, subdir), exist_ok=True)

    # Загрузка данных
    print("Loading MNIST")
    training = get_labeled_data(args.mnist_path, bTrain=True)
    testing = get_labeled_data(args.mnist_path, bTrain=False)
    data = testing if test_mode else training

    # Уравнения нейронов
    v_rest_e, v_rest_i = -65*mV, -60*mV
    v_reset_e, v_reset_i = -65*mV, -45*mV
    v_thresh_e, v_thresh_i = -52*mV, -40*mV
    refrac_e, refrac_i = 5*ms, 2*ms
    neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE + I_synI)/nS)/(100*ms) : volt (unless refractory)
        I_synE = ge * nS * -v : amp
        I_synI = gi * nS * (-100.*mV - v) : amp
        dge/dt = -ge/(1.0*ms) : 1
        dgi/dt = -gi/(2.0*ms) : 1
    '''
    if test_mode:
        neuron_eqs_e += '\n  theta : volt'
    else:
        neuron_eqs_e += '\n  dtheta/dt = -theta/(10*second) : volt'
    neuron_eqs_e += '\n  dtimer/dt = 0.1 : second'
    neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE + I_synI)/nS)/(10*ms) : volt (unless refractory)
        I_synE = ge * nS * -v : amp
        I_synI = gi * nS * (-85.*mV - v) : amp
        dge/dt = -ge/(1.0*ms) : 1
        dgi/dt = -gi/(2.0*ms) : 1
    '''

    # Инициализация сети
    input_group = PoissonGroup(n_input, 0*Hz, name='Xe')
    exc_group = NeuronGroup(n_e, neuron_eqs_e, threshold='(v>(theta - 20*mV + v_thresh_e)) and (timer>refrac_e)',
                           reset='v=v_reset_e; theta+=0.05*mV', refractory=refrac_e, method='euler', name='Ae')
    inh_group = NeuronGroup(n_i, neuron_eqs_i, threshold='v>v_thresh_i',
                           reset='v=v_reset_i', refractory=refrac_i, method='euler', name='Ai')
    exc_group.v, inh_group.v = v_rest_e, v_rest_i
    exc_group.theta = np.zeros(n_e) * mV if test_mode else np.ones(n_e) * 20 * mV

    # Второй слой
    exc_group2 = None
    if n_e2 > 0:
        exc_group2 = NeuronGroup(n_e2, neuron_eqs_e, threshold='(v>(theta - 20*mV + v_thresh_e)) and (timer>refrac_e)',
                                reset='v=v_reset_e; theta+=0.05*mV', refractory=refrac_e, method='euler', name='Ae2')
        exc_group2.v = v_rest_e
        exc_group2.theta = np.zeros(n_e2) * mV if test_mode else np.ones(n_e2) * 20 * mV

    # Соединения
    connections = {}
    weight_path = os.path.join(data_path, 'weights' if test_mode else 'random')
    connections['XeAe'] = Synapses(input_group, exc_group, model='w : 1', on_pre='ge += w', delay=1*ms)
    w_ee = get_matrix_from_file(os.path.join(weight_path, 'XeAe.npy'), (n_input, n_e))
    connections['XeAe'].connect(True)
    connections['XeAe'].w = w_ee.flatten()

    connections['XeAi'] = Synapses(input_group, inh_group, model='w : 1', on_pre='ge += w', delay=1*ms)
    for i, j, w in np.load(os.path.join(weight_path, 'XeAi.npy'), allow_pickle=True):
        i, j = int(i), int(j)
        connections['XeAi'].connect(i=i, j=j)
        connections['XeAi'].w[i, j] = w

    connections['AeAi'] = Synapses(exc_group, inh_group, model='w : 1', on_pre='ge += w', delay=0*ms)
    for i, j, w in np.load(os.path.join(weight_path, 'AeAi.npy'), allow_pickle=True):
        i, j = int(i), int(j)
        connections['AeAi'].connect(i=i, j=j)
        connections['AeAi'].w[i, j] = w

    connections['AiAe'] = Synapses(inh_group, exc_group, model='w : 1', on_pre='gi += w', delay=1*ms)
    for i, j, w in np.load(os.path.join(weight_path, 'AiAe.npy'), allow_pickle=True):
        i, j = int(i), int(j)
        connections['AiAe'].connect(i=i, j=j)
        connections['AiAe'].w[i, j] = w

    if n_e2 > 0:
        connections['AeAe2'] = Synapses(exc_group, exc_group2, model='w : 1', on_pre='ge += w', delay=1*ms)
        w_ee2 = np.random.random((n_e, n_e2)).astype(np.float32) * 0.3
        connections['AeAe2'].connect(True)
        connections['AeAe2'].w = w_ee2.flatten()

    # STDP
    if not test_mode:
        eqs_stdp_ee = '''
            w : 1
            dpre/dt = -pre/(20*ms) : 1 (event-driven)
            dpost1/dt = -post1/(20*ms) : 1 (event-driven)
            dpost2/dt = -post2/(20*ms) : 1 (event-driven)
        '''
        eqs_stdp_pre_ee = 'pre += 0.0001; w = clip(w - post1, 0, 1)'
        eqs_stdp_post_ee = 'post1 += 0.01; post2 += 0.01; w = clip(w + pre, 0, 1)'
        connections['XeAe'].model = eqs_stdp_ee
        connections['XeAe'].on_pre = eqs_stdp_pre_ee
        connections['XeAe'].on_post = eqs_stdp_post_ee
        if n_e2 > 0:
            connections['AeAe2'].model = eqs_stdp_ee
            connections['AeAe2'].on_pre = eqs_stdp_pre_ee
            connections['AeAe2'].on_post = eqs_stdp_post_ee

    # Мониторинг
    spike_monitor_e = SpikeMonitor(exc_group, name='spike_monitor_e')
    spike_monitor_e2 = SpikeMonitor(exc_group2, name='spike_monitor_e2') if n_e2 > 0 else None
    result_monitor = np.zeros((update_interval, n_e2 if n_e2 > 0 else n_e), dtype=np.float32)
    input_numbers = []
    performance = []

    # Сеть
    net = Network(input_group, exc_group, inh_group, *connections.values(), spike_monitor_e)
    if n_e2 > 0:
        net.add(exc_group2, spike_monitor_e2)
    net.store()

    # Восстановление из чекпоинта
    if args.resume_from is not None:
        checkpoint_file = os.path.join(save_path, 'checkpoints', f'checkpoint_{args.resume_from}.b2')
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found")
        net.restore('checkpoint', filename=checkpoint_file)
        j = args.resume_from
        input_numbers = list(np.load(os.path.join(save_path, 'activity', f'input_numbers_{j}.npy')))
        result_monitor = np.load(os.path.join(save_path, 'activity', f'result_monitor_{j}.npy'))
        previous_spike_count = np.bincount(spike_monitor_e.i, minlength=n_e)[:n_e].astype(np.float32)
        if n_e2 > 0:
            previous_spike_count = np.bincount(spike_monitor_e2.i, minlength=n_e2)[:n_e2].astype(np.float32)
    else:
        j = 0
        previous_spike_count = np.zeros(n_e2 if n_e2 > 0 else n_e, dtype=np.float32)

    # Тренировка/тестирование
    print(f"Starting {'testing' if test_mode else 'training'}")
    while j < num_examples:
        rates = data['X'][j % len(data['X'])] / 8. * input_intensity * Hz
        input_group.rates = rates
        net.run(single_example_time)
        net.run(resting_time)

        # Обновление мониторов
        current_spike_count = np.bincount(spike_monitor_e2.i, minlength=n_e2)[:n_e2].astype(np.float32) if n_e2 > 0 else \
                             np.bincount(spike_monitor_e.i, minlength=n_e)[:n_e].astype(np.float32)
        spike_diff = current_spike_count - previous_spike_count
        previous_spike_count = current_spike_count.copy()

        if np.sum(spike_diff) < 5:
            input_intensity += 1
            input_group.rates = 0 * Hz
            net.run(resting_time)
            continue

        result_monitor[j % update_interval] = spike_diff
        input_numbers.append(data['y'][j % len(data['y'])])
        j += 1

        # Обновление весов и назначений
        if j % weight_update_interval == 0 and not test_mode:
            normalize_weights(connections, weight_sum, 'XeAe')
            if n_e2 > 0:
                normalize_weights(connections, weight_sum, 'AeAe2')

        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(result_monitor, input_numbers[-update_interval:], n_e2 if n_e2 > 0 else n_e)
            predictions = get_recognized_number_ranking(assignments, result_monitor)
            correct = np.sum(predictions[0] == np.array(input_numbers[-update_interval:]))
            accuracy = correct / update_interval * 100
            performance.append(accuracy)
            print(f"Iteration {j}, Accuracy: {accuracy:.2f}%, Memory: {psutil.virtual_memory().percent}%")
            result_monitor.fill(0)
            if n_e2 > 0:
                spike_monitor_e2.i, spike_monitor_e2.t = [], []
            else:
                spike_monitor_e.i, spike_monitor_e.t = [], []
            gc.collect()

        # Сохранение результатов
        if j % save_interval == 0 and j > 0:
            net.store('checkpoint', filename=os.path.join(save_path, 'checkpoints', f'checkpoint_{j}.b2'))
            np.save(os.path.join(save_path, 'activity', f'result_monitor_{j}.npy'), result_monitor)
            np.save(os.path.join(save_path, 'activity', f'input_numbers_{j}.npy'), np.array(input_numbers, dtype=np.int32))
            if not test_mode:
                np.save(os.path.join(save_path, 'weights', f'XeAe_{j}.npy'), connections['XeAe'].w)
                np.save(os.path.join(save_path, 'weights', f'theta_{j}.npy'), exc_group.theta[:])
                if n_e2 > 0:
                    np.save(os.path.join(save_path, 'weights', f'AeAe2_{j}.npy'), connections['AeAe2'].w)
                    np.save(os.path.join(save_path, 'weights', f'theta2_{j}.npy'), exc_group2.theta[:])
            plot_2d_input_weights(connections['XeAe'].w, n_input, n_e, save_path, 'layer1')
            if n_e2 > 0:
                plot_2d_input_weights(connections['AeAe2'].w, n_e, n_e2, save_path, 'layer2')

    # Финальное сохранение
    np.save(os.path.join(save_path, 'activity', 'result_monitor_final.npy'), result_monitor)
    np.save(os.path.join(save_path, 'activity', 'input_numbers_final.npy'), np.array(input_numbers, dtype=np.int32))
    if not test_mode:
        np.save(os.path.join(save_path, 'weights', 'XeAe_final.npy'), connections['XeAe'].w)
        np.save(os.path.join(save_path, 'weights', 'theta_final.npy'), exc_group.theta[:])
        if n_e2 > 0:
            np.save(os.path.join(save_path, 'weights', 'AeAe2_final.npy'), connections['AeAe2'].w)
            np.save(os.path.join(save_path, 'weights', 'theta2_final.npy'), exc_group2.theta[:])

    # Визуализация производительности
    plt.plot(performance)
    plt.title('Training Accuracy' if not test_mode else 'Testing Accuracy')
    plt.xlabel('Update Interval')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(save_path, 'accuracy.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN on MNIST with Brian2")
    parser.add_argument('--n_e', type=int, default=400, help="Number of excitatory neurons in first layer")
    parser.add_argument('--n_e2', type=int, default=200, help="Number of excitatory neurons in second layer")
    parser.add_argument('--example_time', type=float, default=0.35, help="Time per example (s)")
    parser.add_argument('--resting_time', type=float, default=0.15, help="Resting time (s)")
    parser.add_argument('--num_examples', type=int, default=60000, help="Number of examples")
    parser.add_argument('--update_interval', type=int, default=1000, help="Update interval")
    parser.add_argument('--weight_update_interval', type=int, default=100, help="Weight update interval")
    parser.add_argument('--save_interval', type=int, default=10000, help="Save interval")
    parser.add_argument('--input_intensity', type=float, default=2.0, help="Input intensity")
    parser.add_argument('--weight_sum', type=float, default=78.0, help="Target weight sum")
    parser.add_argument('--num_threads', type=int, default=4, help="Number of CPU threads")
    parser.add_argument('--resume_from', type=int, default=None, help="Resume from checkpoint iteration")
    parser.add_argument('--mnist_path', type=str, default='./mnist/', help="Path to MNIST")
    parser.add_argument('--data_path', type=str, default='./', help="Path to weights")
    parser.add_argument('--save_path', type=str, default='./results/', help="Path to save results")
    parser.add_argument('--test_mode', action='store_true', help="Run in test mode")
    args = parser.parse_args()
    main(args)
