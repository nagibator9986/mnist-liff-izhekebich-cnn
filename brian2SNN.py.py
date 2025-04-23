# -*- coding: utf-8 -*-
from brian2 import *
import numpy as np
from sklearn.datasets import fetch_openml
import time
import os

# Установка seed для воспроизводимости
np.random.seed(42)

# Настройка устройства для принудительного UTF-8
set_device('runtime', build_options={'force_encoding': 'utf-8'})

# Устанавливаем глобальный dt
defaultclock.dt = 0.1 * ms

# --- Шаг 1: Загрузка и предобработка MNIST ---
print("Загрузка датасета MNIST...")
start_time = time.time()
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"] / 255.0
y = mnist["target"].astype(np.int32)

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:70000]
y_test = y[60000:70000]

del mnist
print(f"Загрузка данных завершена за {time.time() - start_time:.2f} секунд")

# --- Шаг 2: Параметры сети ---
N_input = 784
N_excitatory = 1600
N_inhibitory = N_excitatory
max_rate = 20 * Hz

taum = 100 * ms
Ee = 0 * mV
Ei = -100 * mV
El = -70 * mV
v_thresh = -55 * mV
v_reset = -60 * mV
taue = 5 * ms
taui = 10 * ms
tau_theta = 100 * second
delta_theta = 0.01 * mV

tau_pre = 20 * ms
tau_post = 20 * ms
etaPre = 0.01
etaPost = -0.005
w_max = 1.0

duration = 350 * ms
rest_duration = 150 * ms

# --- Шаг 3: Определение групп нейронов ---
start_time = time.time()

# Используем PoissonGroup
input_group = PoissonGroup(N_input, rates=0 * Hz)

eqs_excitatory = '''
dv/dt = (ge*(Ee - v) + gi*(Ei - v) + El - v)/taum : volt
dge/dt = -ge / taue : 1
dgi/dt = -gi / taui : 1
dtheta/dt = -theta / tau_theta : volt
'''
excitatory_group = NeuronGroup(N_excitatory, eqs_excitatory, threshold='v > v_thresh + theta', 
                               reset='v = v_reset; theta += delta_theta', method='euler')

eqs_inhibitory = '''
dv/dt = (ge*(Ee - v) + El - v)/taum : volt
dge/dt = -ge / taue : 1
'''
inhibitory_group = NeuronGroup(N_inhibitory, eqs_inhibitory, threshold='v > v_thresh', 
                              reset='v = v_reset', method='euler')

excitatory_group.v = El
excitatory_group.ge = 0
excitatory_group.gi = 0
excitatory_group.theta = 0 * mV
inhibitory_group.v = El
inhibitory_group.ge = 0

# --- Шаг 4: Определение синапсов ---
stdp = Synapses(input_group, excitatory_group,
                model='''
                w : 1
                pre_trace : 1
                post_trace : 1
                etaPre : 1 (constant)
                etaPost : 1 (constant)
                wmax : 1 (constant)
                pre_count : 1
                post_count : 1
                ''',
                on_pre='''
                ge_post += w * 10
                pre_trace = clip(pre_trace - 0.1, 0, 1000)
                pre_trace += 1
                pre_count += 1
                w = clip(w + etaPre * post_trace, 0, wmax)
                ''',
                on_post='''
                post_trace = clip(post_trace - 0.1, 0, 1000)
                post_trace += 1
                post_count += 1
                w = clip(w + etaPost * pre_trace, 0, wmax)
                ''')
stdp.connect()
stdp.w = np.random.uniform(0, 0.5, size=len(stdp))
stdp.etaPre = etaPre
stdp.etaPost = etaPost
stdp.wmax = w_max
stdp.pre_count = 0
stdp.post_count = 0
stdp.pre_trace = 0
stdp.post_trace = 0

# Отладка: проверяем, сколько синапсов подключено
print(f"Количество подключённых синапсов: {len(stdp)}")

print(f"Начальные веса: средний w = {np.mean(stdp.w):.4f}, максимальный w = {np.max(stdp.w):.4f}")

syn_excitatory_inhibitory = Synapses(excitatory_group, inhibitory_group, on_pre='ge_post += 10.0')
syn_excitatory_inhibitory.connect(j='i')

syn_inhibitory_excitatory = Synapses(inhibitory_group, excitatory_group, on_pre='gi_post += -5.0')
syn_inhibitory_excitatory.connect(condition='i != j')

print(f"Инициализация сети завершена за {time.time() - start_time:.2f} секунд")

# --- Шаг 5: Функции ---
def reset_network():
    excitatory_group.v = El
    inhibitory_group.v = El
    excitatory_group.ge = 0
    excitatory_group.gi = 0
    inhibitory_group.ge = 0

def set_input_rates(image):
    rates = image * max_rate
    input_group.rates = rates
    max_rate_input = np.max(rates)
    if max_rate_input > 0:
        print(f"Максимальная входная частота: {max_rate_input/Hz:.2f} Hz")

# --- Шаг 6: Обучение сети ---
num_samples = 60000  # Полный тренировочный набор
num_epochs = 30  # Как в вашем успешном эксперименте
namespace = {
    'taum': taum, 'Ee': Ee, 'Ei': Ei, 'El': El, 'v_thresh': v_thresh,
    'v_reset': v_reset, 'taue': taue, 'taui': taui, 'tau_theta': tau_theta,
    'delta_theta': delta_theta
}

# Создаём файл для логов
log_file = "training_log.txt"
if os.path.exists(log_file):
    os.remove(log_file)

spike_mon_train = SpikeMonitor(excitatory_group)
spike_mon_input = SpikeMonitor(input_group)
state_mon_excitatory = StateMonitor(excitatory_group, ['v', 'ge'], record=[0, 100, 500])
state_mon_stdp = StateMonitor(stdp, ['w', 'pre_trace', 'post_trace', 'pre_count', 'post_count'], record=[0, 1000, 10000])

print("Начало обучения...")
start_time = time.time()
for epoch in range(num_epochs):
    print(f"Эпоха {epoch+1}/{num_epochs}")
    for idx in range(num_samples):
        if idx % 1000 == 0:
            print(f"Тренировочный образец {idx}/{num_samples}")
        image = X_train[idx]
        set_input_rates(image)
        reset_network()
        run(duration, namespace=namespace)
        set_input_rates(np.zeros(N_input))
        run(rest_duration, namespace=namespace)
    
    # Логирование после каждой эпохи
    total_spikes = np.sum(spike_mon_train.count)
    input_spikes = np.sum(spike_mon_input.count)
    max_v = np.max(state_mon_excitatory.v/mV)
    max_ge = np.max(state_mon_excitatory.ge)
    min_w = np.min(state_mon_stdp.w)
    max_pre_trace = np.max(state_mon_stdp.pre_trace)
    max_post_trace = np.max(state_mon_stdp.post_trace)
    pre_count = np.max(state_mon_stdp.pre_count)
    post_count = np.max(state_mon_stdp.post_count)
    mean_w = np.mean(stdp.w)
    max_w = np.max(stdp.w)
    mean_theta = np.mean(excitatory_group.theta)
    max_theta = np.max(excitatory_group.theta)
    
    log_message = (f"Эпоха {epoch+1}/{num_epochs}\n"
                   f"Всего спайков (excitatory): {total_spikes}\n"
                   f"Всего спайков (input): {input_spikes}\n"
                   f"Максимальный потенциал v: {max_v:.2f} mV\n"
                   f"Максимальная проводимость ge: {max_ge:.4f}\n"
                   f"Минимальный вес w: {min_w:.4f}\n"
                   f"Максимальный pre_trace: {max_pre_trace:.2f}\n"
                   f"Максимальный post_trace: {max_post_trace:.2f}\n"
                   f"Всего вызовов on_pre: {pre_count:.0f}\n"
                   f"Всего вызовов on_post: {post_count:.0f}\n"
                   f"Средний вес w: {mean_w:.4f}, максимальный вес w: {max_w:.4f}\n"
                   f"Среднее theta: {mean_theta/mV:.2f} mV, максимальное theta: {max_theta/mV:.2f} mV\n\n")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)
    print(log_message)

print(f"Обучение завершено за {time.time() - start_time:.2f} секунд")

# Сбрасываем theta перед присвоением меток
excitatory_group.theta = 0 * mV
print("Theta сброшено перед присвоением меток")

# --- Шаг 7: Присвоение меток нейронам ---
print("Присвоение меток нейронам...")
num_classes = 10
labeling_samples = 1000  # Увеличим для точности
firing_rates_sum = np.zeros((N_excitatory, num_classes), dtype=np.float32)
counts = np.zeros(num_classes, dtype=np.int32)

spike_mon_excitatory = SpikeMonitor(excitatory_group)

class_images = [[] for _ in range(num_classes)]
start_time = time.time()
for idx in range(labeling_samples):
    label = y_train[idx]
    class_images[label].append(X_train[idx])

for c in range(num_classes):
    indices = class_images[c]
    counts[c] = len(indices)
    print(f"Класс {c}: {counts[c]} образцов")
    for image in indices:
        set_input_rates(image)
        reset_network()
        spikes_before = spike_mon_excitatory.count[:]
        run(duration, namespace=namespace)
        spikes_after = spike_mon_excitatory.count[:]
        firing_rates = (spikes_after - spikes_before) / duration
        firing_rates_sum[:, c] += firing_rates
        if np.sum(firing_rates) == 0:
            print(f"Предупреждение: частота спайков для образца класса {c} равна 0!")

counts[counts == 0] = 1
average_firing_rates = firing_rates_sum / counts
neuron_assignments = np.argmax(average_firing_rates, axis=1)

unique, counts_assigned = np.unique(neuron_assignments, return_counts=True)
print("Распределение нейронов по классам после присвоения меток:")
for cls, count in zip(unique, counts_assigned):
    print(f"Класс {cls}: {count} нейронов")

del firing_rates_sum, average_firing_rates
print(f"Присвоение меток завершено за {time.time() - start_time:.2f} секунд")

# --- Шаг 8: Тестирование ---
print("Оценка на тестовом наборе...")
correct = 0
start_time = time.time()
for idx in range(len(X_test)):
    if idx % 1000 == 0:
        print(f"Тестовый образец {idx}/{len(X_test)}")
    image = X_test[idx]
    true_label = y_test[idx]
    set_input_rates(image)
    reset_network()
    spikes_before = spike_mon_excitatory.count[:]
    run(duration, namespace=namespace)
    spikes_after = spike_mon_excitatory.count[:]
    firing_rates = (spikes_after - spikes_before) / duration
    class_firing_rates = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        neurons_c = np.where(neuron_assignments == c)[0]
        if len(neurons_c) > 0:
            class_firing_rates[c] = np.mean(firing_rates[neurons_c])
    predicted_label = np.argmax(class_firing_rates)
    if predicted_label == true_label:
        correct += 1

accuracy = correct / len(X_test)
print(f"Точность на тесте: {accuracy * 100:.2f}%")
print(f"Тестирование завершено за {time.time() - start_time:.2f} секунд")

# Сохраняем финальный результат в лог
with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"Точность на тесте: {accuracy * 100:.2f}%\n")

# Освобождаем память
del X_train, y_train, X_test, y_test