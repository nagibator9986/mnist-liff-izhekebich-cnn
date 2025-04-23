import numpy as np
import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Параметры модели
INPUT_SIZE = 784        # Размер входных данных MNIST (28x28)
HIDDEN_SIZE = 2048      # Увеличенный скрытый слой для лучшего обучения
OUTPUT_SIZE = 10        # 10 классов цифр
TIME_STEPS = 20         # Временные шаги симуляции
EPOCHS = 30             
BATCH_SIZE = 128        
TDP = 160               # TDP RTX 4060 Ti (Вт)
STDP_LR = 0.01          # Скорость обучения STDP
HEBBIAN_LR = 0.005      # Скорость обучения Hebbian
STDP_TAU_PLUS = 10.0    # Уменьшенные временные константы
STDP_TAU_MINUS = 10.0   
STDP_A_PLUS = 0.015     # Оптимизированные амплитуды STDP
STDP_A_MINUS = 0.015    
WEIGHT_DECAY = 1e-4     # Слабый L2-регуляризатор

class IzhikevichSNN:
    def __init__(self):
        # Инициализация весов
        self.w_input = cp.random.normal(0, 0.5, (INPUT_SIZE, HIDDEN_SIZE)) * 0.1
        self.w_hidden = cp.random.normal(0, 0.5, (HIDDEN_SIZE, OUTPUT_SIZE)) * 0.1
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Оптимизированные параметры Ижикевича (фазический режим)
        self.neuron_params = {
            'a': 0.1,       # Быстрое восстановление
            'b': 0.2,       # Чувствительность к потенциалу
            'c': -55.0,     # Потенциал после спайка
            'd': 4.0,       # Адаптация после спайка
            'threshold': 25.0  # Порог срабатывания
        }
        
        # История обучения
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'spike_rate': [],
            'time_per_epoch': [],
            'energy_per_epoch': []
        }

        # Создание файла для логирования
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'training_metrics_{timestamp}.csv'
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Acc', 'Val Acc', 'Spikes/neuron', 
                            'Time (s)', 'Energy (Wh)', 'Mean I_h', 'Input Spikes', 
                            'Mean Outputs', 'Var w_input', 'Var w_hidden'])

    def izhikevich_update(self, v, u, I):
        """Обновление состояния нейронов Ижикевича."""
        dv = 0.04 * v**2 + 5 * v + 140 - u + I
        du = self.neuron_params['a'] * (self.neuron_params['b'] * v - u)
        v += dv * 0.5  # Euler integration
        u += du * 0.5
        spikes = (v >= self.neuron_params['threshold']).astype(cp.float32)
        v = cp.where(spikes, self.neuron_params['c'], v)
        u = cp.where(spikes, u + self.neuron_params['d'], u)
        return v, u, spikes

    def encode_input(self, x):
        """Кодирование входных данных в спайки (популяционное кодирование)."""
        x = cp.asarray(x)
        spike_prob = cp.repeat(x[None, :, :], TIME_STEPS, axis=0)
        spike_train = cp.random.rand(*spike_prob.shape) < spike_prob
        return spike_train.astype(cp.float32)

        def stdp_update(self, pre_spikes, post_spikes, weights):
            """Ускоренная версия STDP с экспоненциальным окном."""
            dw = cp.zeros_like(weights)
            for t in range(1, TIME_STEPS):
                # LTP: если пре-синаптический нейрон сработал до пост-синаптического
                dw += STDP_A_PLUS * cp.dot(pre_spikes[t-1].T, post_spikes[t])
                # LTD: если пост-синаптический нейрон сработал раньше
                dw -= STDP_A_MINUS * cp.dot(pre_spikes[t].T, post_spikes[t-1])
        weights += STDP_LR * dw / TIME_STEPS
        weights = cp.clip(weights, -1.0, 1.0)
        # Нормализация весов
        norm = cp.linalg.norm(weights, axis=0, keepdims=True)
        weights = cp.where(norm > 1.0, weights / norm, weights)
        return weights

    def hebbian_update(self, spikes_h, outputs, targets, weights):
        """Геббовское обучение с учителем."""
        error = targets - outputs
        dw = cp.dot(spikes_h.T, error) / spikes_h.shape[0]
        weights += HEBBIAN_LR * dw
        weights -= WEIGHT_DECAY * weights  # L2-регуляризация
        return cp.clip(weights, -1.0, 1.0)

    def forward(self, x, train_mode=False):
        """Прямой проход с возможностью обучения."""
        batch_size = x.shape[0]
        # Инициализация состояний нейронов
        v_h = cp.full((batch_size, HIDDEN_SIZE), -65.0, dtype=cp.float32)
        u_h = cp.zeros((batch_size, HIDDEN_SIZE), dtype=cp.float32)
        v_o = cp.full((batch_size, OUTPUT_SIZE), -65.0, dtype=cp.float32)
        u_o = cp.zeros((batch_size, OUTPUT_SIZE), dtype=cp.float32)
        
        output = cp.zeros((batch_size, OUTPUT_SIZE), dtype=cp.float32)
        spike_count = 0
        spikes_h_sum = cp.zeros((batch_size, HIDDEN_SIZE), dtype=cp.float32)
        spike_train = self.encode_input(x)
        mean_i_h = 0.0
        input_spikes = cp.mean(spike_train).get()
        
        post_spikes_h = cp.zeros((TIME_STEPS, batch_size, HIDDEN_SIZE), dtype=cp.float32)
        
        for t in range(TIME_STEPS):
            # Скрытый слой
            I_h = cp.dot(spike_train[t], self.w_input) * 0.5  # Уменьшенный масштаб
            mean_i_h += cp.mean(cp.abs(I_h)).get() / TIME_STEPS
            v_h, u_h, spikes_h = self.izhikevich_update(v_h, u_h, I_h)
            spikes_h_sum += spikes_h
            spike_count += cp.sum(spikes_h)
            post_spikes_h[t] = spikes_h
            
            # Выходной слой
            I_o = cp.dot(spikes_h, self.w_hidden) * 0.5
            v_o, u_o, spikes_o = self.izhikevich_update(v_o, u_o, I_o)
            output += spikes_o
        
        outputs = output / TIME_STEPS
        mean_outputs = cp.mean(outputs).get()
        
        if train_mode:
            return outputs, spike_count, spikes_h_sum / TIME_STEPS, spike_train, post_spikes_h, mean_i_h, input_spikes, mean_outputs
        return outputs, spike_count, spikes_h_sum / TIME_STEPS

    def train(self, X_train, y_train, X_val, y_val):
        """Обучение модели."""
        X_train_gpu = cp.asarray(self.scaler.fit_transform(X_train), dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
        X_val_gpu = cp.asarray(self.scaler.transform(X_val), dtype=cp.float32)
        y_val = y_val.astype(np.int32)
        
        for epoch in range(EPOCHS):
            start_time = time.time()
            correct = 0
            total_spikes = 0
            total_i_h = 0.0
            total_input_spikes = 0.0
            total_mean_outputs = 0.0
            
            # Перемешивание данных
            indices = cp.random.permutation(len(X_train_gpu))
            X_shuffled = X_train_gpu[indices]
            y_shuffled = y_train_gpu[indices]
            
            # Обучение по батчам
            for i in tqdm(range(0, len(X_shuffled), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}"):
                batch_X = X_shuffled[i:i+BATCH_SIZE]
                batch_y = y_shuffled[i:i+BATCH_SIZE]
                
                # Прямой проход + сбор метрик
                outputs, batch_spikes, spikes_h_batch, pre_spikes, post_spikes_h, mean_i_h, input_spikes, mean_outputs = self.forward(batch_X, train_mode=True)
                total_spikes += batch_spikes
                total_i_h += mean_i_h
                total_input_spikes += input_spikes
                total_mean_outputs += mean_outputs
                
                # STDP для входного слоя
                self.w_input = self.stdp_update(pre_spikes, post_spikes_h, self.w_input)
                
                # Геббовское обучение для выходного слоя
                targets = cp.eye(OUTPUT_SIZE)[batch_y]
                self.w_hidden = self.hebbian_update(spikes_h_batch, outputs, targets, self.w_hidden)
                
                # Расчет точности
                preds = cp.argmax(outputs, axis=1)
                correct += cp.sum(preds == batch_y).get()
            
            # Метрики эпохи
            acc = correct / len(X_shuffled)
            spike_rate = total_spikes / (len(X_shuffled) * HIDDEN_SIZE * TIME_STEPS)
            val_acc = self.evaluate(X_val_gpu, y_val)
            epoch_time = time.time() - start_time
            energy_epoch = TDP * (epoch_time / 3600)
            mean_i_h = total_i_h / (len(X_shuffled) // BATCH_SIZE)
            mean_input_spikes = total_input_spikes / (len(X_shuffled) // BATCH_SIZE)
            mean_outputs = total_mean_outputs / (len(X_shuffled) // BATCH_SIZE)
            var_w_input = cp.var(self.w_input).get()
            var_w_hidden = cp.var(self.w_hidden).get()
            
            # Сохранение истории
            self.history['accuracy'].append(acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['spike_rate'].append(spike_rate.get())
            self.history['time_per_epoch'].append(epoch_time)
            self.history['energy_per_epoch'].append(energy_epoch)
            
            # Логирование
            log_message = (f"Epoch {epoch+1} | Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | "
                          f"Spikes: {spike_rate:.2f}/neuron | Time: {epoch_time:.1f}s | "
                          f"Energy: {energy_epoch:.2f} Wh | Mean I_h: {mean_i_h:.2f} | "
                          f"Input Spikes: {mean_input_spikes:.4f} | Outputs: {mean_outputs:.4f}")
            print(log_message)
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, acc, val_acc, spike_rate.get(), epoch_time, 
                                energy_epoch, mean_i_h, mean_input_spikes, mean_outputs, 
                                var_w_input, var_w_hidden])

    def evaluate(self, X_test, y_test):
        """Оценка точности на тестовых данных."""
        correct = 0
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_X = X_test[i:i+BATCH_SIZE]
            batch_y = y_test[i:i+BATCH_SIZE]
            output, _, _ = self.forward(batch_X)
            preds = cp.argmax(output, axis=1)
            correct += cp.sum(preds == cp.asarray(batch_y)).get()
        return correct / len(X_test)

    def plot_training(self):
        """Визуализация процесса обучения."""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.history['accuracy'], label='Train')
        plt.plot(self.history['val_accuracy'], label='Validation')
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.history['spike_rate'])
        plt.title("Spike Rate (Spikes/neuron)")
        plt.xlabel("Epoch")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.history['time_per_epoch'])
        plt.title("Time per Epoch (s)")
        plt.xlabel("Epoch")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.history['energy_per_epoch'])
        plt.title("Energy per Epoch (Wh)")
        plt.xlabel("Epoch")
        
        plt.tight_layout()
        plt.savefig('training_analysis.png')
        plt.close()

# Инициализация GPU
print("Initializing GPU...")
cp.cuda.Device(0).use()
print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# Загрузка данных MNIST
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int32)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Обучение модели
print("\nTraining SNN with optimized Izhikevich neurons...")
snn = IzhikevichSNN()
snn.train(X_train, y_train, X_val, y_val)

# Оценка на тестовых данных
test_acc = snn.evaluate(snn.scaler.transform(X_test), y_test)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Визуализация результатов
snn.plot_training()