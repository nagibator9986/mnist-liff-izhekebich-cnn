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
    print(f"Opening {file_name}")
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"MNIST file {file_name} not found")
    with gzip.open(file_name, 'rb') as f:
        print(f"Loading pickle data for {file_name}...")
        data = pickle.load(f)
        print(f"Pickle data loaded for {file_name}")
    print(f"Reshaping data for {file_name}...")
    result = {'X': data[0].reshape((-1, 784)).astype(np.float32), 'y': data[1].astype(np.int32)}
    print(f"Data reshaping complete for {file_name}")
    return result

def get_matrix_from_file(file_name, shape):
    """Загружает матрицу весов из файла."""
    print(f"Loading weight file: {file_name}")
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Weight file {file_name} not found")
    data = np.load(file_name, allow_pickle=True)
    print(f"Weight file {file_name} loaded, data dtype: {data.dtype}, shape: {data.shape}")
    print(f"First row: {data[:1]}")
    invalid_indices = [(row['i'], row['j']) for row in data if row['i'] >= shape[0] or row['j'] >= shape[1]]
    if invalid_indices:
        raise ValueError(f"Invalid indices in {file_name}: {invalid_indices[:10]} (showing first 10), expected i < {shape[0]}, j < {shape[1]}")
    matrix = np.zeros(shape, dtype=np.float32)
    for row in data:
        i = int(row['i'])
        j = int(row['j'])
        w = float(row['w'])
        matrix[i, j] = w
    print(f"Matrix for {file_name} created")
    return matrix

def normalize_weights(synapses, weight_sum):
    """Нормализует веса синапсов."""
    # Преобразуем VariableView в массив NumPy
    weights = np.array(synapses.w)
    # Преобразуем в двумерный массив (source.N, target.N)
    weight_matrix = weights.reshape(synapses.source.N, synapses.target.N)
    # Вычисляем суммы по столбцам
    col_sums = np.sum(weight_matrix, axis=0)
    col_sums[col_sums == 0] = 1  # Избегаем деления на ноль
    # Нормализуем веса
    weight_matrix *= weight_sum / col_sums
    # Обновляем веса синапсов
    synapses.w = weight_matrix.flatten()

def get_new_assignments(result_monitor, input_numbers, n_neurons):
    """Назначает классы нейронам на основе активности."""
    assignments = np.ones(n_neurons, dtype=np.int32) * -1
    max_rates = np.zeros(n_neurons, dtype=np.float32)
    input_nums = np.asarray(input_numbers, dtype=np.int32)
    for j in range(10):
        mask = input_nums == j
        num_inputs = np.sum(mask)
        if num_inputs > 0:
            rate = np.sum(result_monitor[mask], axis=0) / num_inputs
            for i in range(n_neurons):
                if rate[i] > max_rates[i]:
                    max_rates[i] = rate[i]
                    assignments[i] = j
    return assignments

def get_recognized_number_ranking(assignments, spike_rates):
    """Предсказывает классы на основе активности."""
    summed_rates = np.zeros(10, dtype=np.float32)
    num_assignments = np.zeros(10, dtype=np.float32)
    for i in range(10):
        mask = assignments == i
        num_assignments[i] = np.sum(mask)
        if num_assignments[i] > 0:
            # Суммируем спайковые скорости по оси примеров для нейронов, назначенных классу i
            summed_rates[i] = np.sum(spike_rates[:, mask]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def plot_2d_input_weights(weights, n_input, n_neurons, save_path, layer_name='layer1'):
    """Визуализирует веса как тепловую карту."""
    weights_2d = weights.reshape((28, 28, n_neurons))
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < min(n_neurons, 100):
            ax.imshow(weights_2d[:, :, i], cmap='hot_r', interpolation='nearest')
            ax.axis('off')
    plt.savefig(os.path.join(save_path, f'weights_{layer_name}.png'))
    plt.close()

def main(args):
    # Настройка Brian2
    print("Configuring Brian2...")
    set_device('runtime')  # Используем runtime для упрощения
    prefs.codegen.target = 'numpy'  # NumPy для совместимости
    defaultclock.dt = 0.5 * ms  # Шаг времени
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

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
    input_intensity = args.input_intensity
    weight_sum = args.weight_sum
    data_path = args.data_path
    save_path = args.save_path
    test_mode = args.test_mode

    # Создание директорий
    print("Creating directories...")
    for subdir in ['', 'weights', 'activity', 'checkpoints']:
        os.makedirs(os.path.join(save_path, subdir), exist_ok=True)

    # Загрузка данных MNIST
    print("Loading MNIST data...")
    training = get_labeled_data(args.mnist_path, bTrain=True)
    testing = get_labeled_data(args.mnist_path, bTrain=False)
    data = testing if test_mode else training
    print("MNIST data loaded")

    # Загрузка весов
    print("Loading weights...")
    weight_path = data_path
    w_ee = get_matrix_from_file(os.path.join(weight_path, 'XeAe.npy'), (n_input, n_e))
    print("XeAe weights loaded")
    w_ei = get_matrix_from_file(os.path.join(weight_path, 'XeAi.npy'), (n_input, n_i))
    print("XeAi weights loaded")
    w_ii = get_matrix_from_file(os.path.join(weight_path, 'AeAi.npy'), (n_e, n_i))
    print("AeAi weights loaded")
    w_ie = get_matrix_from_file(os.path.join(weight_path, 'AiAe.npy'), (n_i, n_e))
    print("AiAe weights loaded")

    # Уравнения нейронов
    print("Defining neuron equations...")
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
    print("Initializing Brian2 network...")
    print("Creating input Poisson group...")
    input_group = PoissonGroup(n_input, 0*Hz, name='Xe')
    print("Creating excitatory neuron group...")
    exc_group = NeuronGroup(n_e, neuron_eqs_e, threshold='(v>(theta - 20*mV + v_thresh_e)) and (timer>refrac_e)',
                           reset='v=v_reset_e; theta+=0.05*mV', refractory=refrac_e, method='euler', name='Ae')
    print("Creating inhibitory neuron group...")
    inh_group = NeuronGroup(n_i, neuron_eqs_i, threshold='v>v_thresh_i',
                           reset='v=v_reset_i', refractory=refrac_i, method='euler', name='Ai')
    exc_group.v, inh_group.v = v_rest_e, v_rest_i
    exc_group.theta = np.zeros(n_e) * mV if test_mode else np.ones(n_e) * 20 * mV

    # Второй слой (если указан)
    exc_group2 = None
    if n_e2 > 0:
        print("Creating second excitatory neuron group...")
        exc_group2 = NeuronGroup(n_e2, neuron_eqs_e, threshold='(v>(theta - 20*mV + v_thresh_e)) and (timer>refrac_e)',
                                reset='v=v_reset_e; theta+=0.05*mV', refractory=refrac_e, method='euler', name='Ae2')
        exc_group2.v = v_rest_e
        exc_group2.theta = np.zeros(n_e2) * mV if test_mode else np.ones(n_e2) * 20 * mV

    # Определение STDP (для тренировочного режима)
    print("Defining STDP equations...")
    eqs_stdp_ee = '''
        w : 1
        dpre/dt = -pre/(20*ms) : 1 (event-driven)
        dpost1/dt = -post1/(20*ms) : 1 (event-driven)
        dpost2/dt = -post2/(20*ms) : 1 (event-driven)
    '''
    eqs_stdp_pre_ee = 'pre += 0.0001; w = clip(w - post1, 0, 1)'
    eqs_stdp_post_ee = 'post1 += 0.01; post2 += 0.01; w = clip(w + pre, 0, 1)'

    # Создание синапсов
    print("Creating synapses...")
    connections = {}
    
    # XeAe синапсы
    print("Connecting XeAe synapses...")
    if test_mode:
        connections['XeAe'] = Synapses(input_group, exc_group, model='w : 1', on_pre='ge += w', delay=1*ms, name='XeAe')
    else:
        connections['XeAe'] = Synapses(input_group, exc_group, model=eqs_stdp_ee, 
                                       on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee, delay=1*ms, name='XeAe')
    i_ee, j_ee = np.nonzero(w_ee)
    connections['XeAe'].connect(i=i_ee, j=j_ee)
    connections['XeAe'].w = w_ee[i_ee, j_ee]

    # XeAi синапсы
    print("Connecting XeAi synapses...")
    connections['XeAi'] = Synapses(input_group, inh_group, model='w : 1', on_pre='ge += w', delay=1*ms, name='XeAi')
    i_ei, j_ei = np.nonzero(w_ei)
    connections['XeAi'].connect(i=i_ei, j=j_ei)
    connections['XeAi'].w = w_ei[i_ei, j_ei]

    # AeAi синапсы
    print("Connecting AeAi synapses...")
    connections['AeAi'] = Synapses(exc_group, inh_group, model='w : 1', on_pre='ge += w', delay=0*ms, name='AeAi')
    i_ii, j_ii = np.nonzero(w_ii)
    connections['AeAi'].connect(i=i_ii, j=j_ii)
    connections['AeAi'].w = w_ii[i_ii, j_ii]

    # AiAe синапсы
    print("Connecting AiAe synapses...")
    connections['AiAe'] = Synapses(inh_group, exc_group, model='w : 1', on_pre='gi += w', delay=1*ms, name='AiAe')
    i_ie, j_ie = np.nonzero(w_ie)
    connections['AiAe'].connect(i=i_ie, j=j_ie)
    connections['AiAe'].w = w_ie[i_ie, j_ie]

    # AeAe2 синапсы (если второй слой активен)
    if n_e2 > 0:
        print("Creating AeAe2 synapses...")
        if test_mode:
            connections['AeAe2'] = Synapses(exc_group, exc_group2, model='w : 1', on_pre='ge += w', delay=1*ms, name='AeAe2')
        else:
            connections['AeAe2'] = Synapses(exc_group, exc_group2, model=eqs_stdp_ee, 
                                            on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee, delay=1*ms, name='AeAe2')
        w_ee2 = np.random.random((n_e, n_e2)).astype(np.float32) * 0.3
        connections['AeAe2'].connect(True)
        connections['AeAe2'].w = w_ee2.flatten()

    # Мониторинг
    print("Setting up monitors...")
    spike_monitor_e = SpikeMonitor(exc_group, name='spike_monitor_e')
    spike_monitor_e2 = SpikeMonitor(exc_group2, name='spike_monitor_e2') if n_e2 > 0 else None
    result_monitor = np.zeros((update_interval, n_e2 if n_e2 > 0 else n_e), dtype=np.float32)
    input_numbers = []
    performance = []

    # Создание сети
    print("Creating network...")
    net = Network(input_group, exc_group, inh_group, *connections.values(), spike_monitor_e)
    if n_e2 > 0:
        net.add(exc_group2, spike_monitor_e2)
    net.store()

    # Восстановление из чекпоинта
    if args.resume_from is not None:
        print(f"Restoring from checkpoint {args.resume_from}...")
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

    # Основной цикл
    print(f"Starting {'testing' if test_mode else 'training'}...")
    while j < num_examples:
        print(f"Iteration {j}/{num_examples}, Memory: {psutil.virtual_memory().percent}%")
        rates = data['X'][j % len(data['X'])] / 8. * input_intensity * Hz
        input_group.rates = rates
        print(f"Running simulation for example {j}...")
        net.run(single_example_time, report='text')
        print(f"Running resting period...")
        input_group.rates = 0 * Hz
        net.run(resting_time)

        # Обновление мониторов
        print("Updating spike counts...")
        current_spike_count = np.bincount(spike_monitor_e2.i, minlength=n_e2)[:n_e2].astype(np.float32) if n_e2 > 0 else \
                             np.bincount(spike_monitor_e.i, minlength=n_e)[:n_e].astype(np.float32)
        spike_diff = current_spike_count - previous_spike_count
        previous_spike_count = current_spike_count.copy()

        if np.sum(spike_diff) < 5:
            print(f"Low spike activity ({np.sum(spike_diff)}), increasing input intensity to {input_intensity + 1}")
            input_intensity += 1
            input_group.rates = 0 * Hz
            net.run(resting_time)
            continue

        result_monitor[j % update_interval] = spike_diff
        input_numbers.append(data['y'][j % len(data['y'])])
        j += 1

        # Обновление весов и назначений
        if j % weight_update_interval == 0 and not test_mode:
            print("Normalizing weights...")
            normalize_weights(connections['XeAe'], weight_sum)
            if n_e2 > 0:
                normalize_weights(connections['AeAe2'], weight_sum)

        if j % update_interval == 0 and j > 0:
            print("Updating assignments and performance...")
            assignments = get_new_assignments(result_monitor, input_numbers[-update_interval:], n_e2 if n_e2 > 0 else n_e)
            predictions = get_recognized_number_ranking(assignments, result_monitor)
            correct = np.sum(predictions[0] == np.array(input_numbers[-update_interval:]))
            accuracy = correct / update_interval * 100
            performance.append(accuracy)
            print(f"Iteration {j}, Accuracy: {accuracy:.2f}%")
            result_monitor.fill(0)
            if n_e2 > 0:
                spike_monitor_e2.i, spike_monitor_e2.t = [], []
            else:
                spike_monitor_e.i, spike_monitor_e.t = [], []
            gc.collect()

        # Сохранение результатов
        if j % save_interval == 0 and j > 0:
            print(f"Saving checkpoint at iteration {j}...")
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
    print("Saving final results...")
    np.save(os.path.join(save_path, 'activity', 'result_monitor_final.npy'), result_monitor)
    np.save(os.path.join(save_path, 'activity', 'input_numbers_final.npy'), np.array(input_numbers, dtype=np.int32))
    if not test_mode:
        np.save(os.path.join(save_path, 'weights', 'XeAe_final.npy'), connections['XeAe'].w)
        np.save(os.path.join(save_path, 'weights', 'theta_final.npy'), exc_group.theta[:])
        if n_e2 > 0:
            np.save(os.path.join(save_path, 'weights', 'AeAe2_final.npy'), connections['AeAe2'].w)
            np.save(os.path.join(save_path, 'weights', 'theta2_final.npy'), exc_group2.theta[:])

    # Визуализация производительности
    print("Plotting performance...")
    plt.plot(performance)
    plt.title('Training Accuracy' if not test_mode else 'Testing Accuracy')
    plt.xlabel('Update Interval')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(save_path, 'accuracy.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN on MNIST with Brian2")
    parser.add_argument('--n_e', type=int, default=200, help="Number of excitatory neurons in first layer")
    parser.add_argument('--n_e2', type=int, default=0, help="Number of excitatory neurons in second layer")
    parser.add_argument('--example_time', type=float, default=0.35, help="Time per example (s)")
    parser.add_argument('--resting_time', type=float, default=0.15, help="Resting time (s)")
    parser.add_argument('--num_examples', type=int, default=10000, help="Number of examples")
    parser.add_argument('--update_interval', type=int, default=1000, help="Update interval")
    parser.add_argument('--weight_update_interval', type=int, default=100, help="Weight update interval")
    parser.add_argument('--save_interval', type=int, default=5000, help="Save interval")
    parser.add_argument('--input_intensity', type=float, default=2.0, help="Input intensity")
    parser.add_argument('--weight_sum', type=float, default=78.0, help="Target weight sum")
    parser.add_argument('--num_threads', type=int, default=2, help="Number of CPU threads")
    parser.add_argument('--resume_from', type=int, default=None, help="Resume from checkpoint iteration")
    parser.add_argument('--mnist_path', type=str, default='./mnist/', help="Path to MNIST")
    parser.add_argument('--data_path', type=str, default='./random/', help="Path to weights")
    parser.add_argument('--save_path', type=str, default='./results/', help="Path to save results")
    parser.add_argument('--test_mode', action='store_true', help="Run in test mode")
    args = parser.parse_args()
    main(args)
