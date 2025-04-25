import numpy as np
import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import csv

# Параметры
INPUT_SIZE = 784
HIDDEN_SIZE = 1024
OUTPUT_SIZE = 10
TIME_STEPS = 50
EPOCHS = 30
BATCH_SIZE = 128
INIT_LR = 0.02
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 1e-4
TDP = 160  # Вт, TDP RTX 4060 Ti

class IzhikevichSNN:
    def __init__(self):
        self.w_input = cp.random.normal(0, 0.5, (INPUT_SIZE, HIDDEN_SIZE)) * 0.3
        self.w_hidden = cp.random.normal(0, 0.5, (HIDDEN_SIZE, OUTPUT_SIZE)) * 0.3
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.m_input = cp.zeros_like(self.w_input)
        self.v_input = cp.zeros_like(self.w_input)
        self.m_hidden = cp.zeros_like(self.w_hidden)
        self.v_hidden = cp.zeros_like(self.w_hidden)
        
        self.neuron_params = {
            'a': 0.02,
            'b': 0.2,
            'c': -65.0,
            'd': 8.0,
            'threshold': 30.0
        }
        
        self.history = {
            'accuracy': [],
            'loss': [],
            'spike_rate': [],
            'val_accuracy': [],
            'time_per_epoch': [],
            'energy_per_epoch': []
        }

        # Инициализация CSV
        with open('training_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Acc', 'Val Acc', 'Loss', 'Spikes/neuron', 'Time (s)', 'Energy (Wh)'])

    def izhikevich_update(self, v, u, I):
        I = cp.clip(I, -1500, 1500)
        dv = 0.04 * v**2 + 5 * v + 140 - u + I
        du = self.neuron_params['a'] * (self.neuron_params['b'] * v - u)
        v += dv * 0.5
        u += du * 0.5
        spikes = (v >= self.neuron_params['threshold']).astype(cp.float32)
        v = cp.where(spikes, self.neuron_params['c'], v)
        u = cp.where(spikes, u + self.neuron_params['d'], u)
        return v, u, spikes

    def surrogate_gradient(self, v):
        return 2.0 / (1.0 + cp.abs(v - self.neuron_params['threshold']))**2

    def encode_input(self, x):
        x = cp.asarray(x)
        spike_train = cp.random.rand(TIME_STEPS, x.shape[0], INPUT_SIZE) < x
        return spike_train.astype(cp.float32)

    def forward(self, x):
        batch_size = x.shape[0]
        v_h = cp.full((batch_size, HIDDEN_SIZE), -65.0, dtype=cp.float32)
        u_h = cp.zeros((batch_size, HIDDEN_SIZE), dtype=cp.float32)
        v_o = cp.full((batch_size, OUTPUT_SIZE), -65.0, dtype=cp.float32)
        u_o = cp.zeros((batch_size, OUTPUT_SIZE), dtype=cp.float32)
        
        output = cp.zeros((batch_size, OUTPUT_SIZE), dtype=cp.float32)
        spike_count = 0
        spikes_h_sum = cp.zeros((batch_size, HIDDEN_SIZE), dtype=cp.float32)
        spike_train = self.encode_input(x)
        
        for t in range(TIME_STEPS):
            I_h = cp.dot(spike_train[t], self.w_input) * 20
            v_h, u_h, spikes_h = self.izhikevich_update(v_h, u_h, I_h)
            spikes_h = spikes_h * (cp.random.rand(*spikes_h.shape) > 0.3)
            spikes_h_sum += spikes_h
            spike_count += cp.sum(spikes_h)
            
            I_o = cp.dot(spikes_h, self.w_hidden) * 20
            v_o, u_o, spikes_o = self.izhikevich_update(v_o, u_o, I_o)
            output += spikes_o
        
        return output / TIME_STEPS, spike_count, spikes_h_sum / TIME_STEPS

    def train(self, X_train, y_train, X_val, y_val):
        X_train_gpu = cp.asarray(self.scaler.fit_transform(X_train), dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
        X_val_gpu = cp.asarray(self.scaler.transform(X_val), dtype=cp.float32)
        y_val = y_val.astype(np.int32)
        
        for epoch in range(EPOCHS):
            start_time = time.time()
            correct = 0
            total_loss = 0
            total_spikes = 0
            t = epoch + 1
            
            indices = cp.random.permutation(len(X_train))
            X_shuffled = X_train_gpu[indices]
            y_shuffled = y_train_gpu[indices]
            
            for i in tqdm(range(0, len(X_shuffled), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}"):
                batch_X = X_shuffled[i:i+BATCH_SIZE]
                batch_y = y_shuffled[i:i+BATCH_SIZE]
                
                outputs, batch_spikes, spikes_h_batch = self.forward(batch_X)
                total_spikes += batch_spikes
                
                targets = cp.eye(OUTPUT_SIZE)[batch_y]
                probs = cp.exp(outputs) / cp.sum(cp.exp(outputs), axis=1, keepdims=True)
                loss = -cp.mean(cp.sum(targets * cp.log(probs + 1e-10), axis=1))
                total_loss += loss.get()
                
                error = probs - targets
                grad_hidden = cp.dot(spikes_h_batch.T, error) / batch_X.shape[0] + WEIGHT_DECAY * self.w_hidden
                self.m_hidden = BETA1 * self.m_hidden + (1 - BETA1) * grad_hidden
                self.v_hidden = BETA2 * self.v_hidden + (1 - BETA2) * (grad_hidden ** 2)
                m_hat = self.m_hidden / (1 - BETA1 ** t)
                v_hat = self.v_hidden / (1 - BETA2 ** t)
                self.w_hidden -= INIT_LR * m_hat / (cp.sqrt(v_hat) + 1e-8)
                self.w_hidden = cp.clip(self.w_hidden, -1, 1)
                
                error_h = cp.dot(error, self.w_hidden.T)
                v_h = cp.full((batch_X.shape[0], HIDDEN_SIZE), -65.0)
                spike_train = self.encode_input(batch_X)
                for t in range(TIME_STEPS):
                    v_h += cp.dot(spike_train[t], self.w_input) * 10
                grad_spikes_h = self.surrogate_gradient(v_h)
                grad_input = cp.dot(batch_X.T, error_h * grad_spikes_h) / batch_X.shape[0] + WEIGHT_DECAY * self.w_input
                self.m_input = BETA1 * self.m_input + (1 - BETA1) * grad_input
                self.v_input = BETA2 * self.v_input + (1 - BETA2) * (grad_input ** 2)
                m_hat = self.m_input / (1 - BETA1 ** t)
                v_hat = self.v_input / (1 - BETA2 ** t)
                self.w_input -= INIT_LR * m_hat / (cp.sqrt(v_hat) + 1e-8)
                self.w_input = cp.clip(self.w_input, -1, 1)
                
                preds = cp.argmax(outputs, axis=1)
                correct += cp.sum(preds == batch_y).get()
            
            acc = correct / len(X_shuffled)
            avg_loss = total_loss / (len(X_shuffled) / BATCH_SIZE)
            spike_rate = total_spikes / (len(X_shuffled) * HIDDEN_SIZE * TIME_STEPS)
            val_acc = self.evaluate(X_val_gpu, y_val)[0]
            epoch_time = time.time() - start_time
            energy_epoch = TDP * (epoch_time / 3600)  # Вт·ч
            
            self.history['accuracy'].append(acc)
            self.history['loss'].append(avg_loss)
            self.history['spike_rate'].append(spike_rate.get() if isinstance(spike_rate, cp.ndarray) else spike_rate)
            self.history['val_accuracy'].append(val_acc)
            self.history['time_per_epoch'].append(epoch_time)
            self.history['energy_per_epoch'].append(energy_epoch)
            
            log_message = (f"Epoch {epoch+1} | Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | "
                          f"Loss: {avg_loss:.4f} | Spikes: {spike_rate:.2f}/neuron | "
                          f"Time: {epoch_time:.1f}s | Energy: {energy_epoch:.2f} Wh")
            print(log_message)
            
            with open('training_metrics.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, acc, val_acc, avg_loss, spike_rate, epoch_time, energy_epoch])

    def evaluate(self, X_test, y_test):
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_X = cp.asarray(X_test[i:i+BATCH_SIZE])
            batch_y = y_test[i:i+BATCH_SIZE]
            output, _, _ = self.forward(batch_X)
            preds = cp.argmax(output, axis=1)
            correct += cp.sum(preds == cp.asarray(batch_y)).get()
            total += batch_y.shape[0]
            all_preds.append(preds.get())
            all_labels.append(batch_y)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return correct / total, all_preds, all_labels

    def post_training_analysis(self, X_train):
        # Общее время и энергия
        total_time = sum(self.history['time_per_epoch'])
        total_energy = sum(self.history['energy_per_epoch'])
        energy_per_sample = total_energy / len(X_train)
        print(f"\nPost-Training Analysis:")
        print(f"Total Training Time: {total_time:.1f} s")
        print(f"Total Energy Used: {total_energy:.2f} Wh")
        print(f"Energy per Sample: {energy_per_sample:.6f} Wh/sample")
        
        # FLOPs
        flops_per_sample = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE) * TIME_STEPS * 2  # Forward + Backward
        total_flops = flops_per_sample * (len(X_train) // BATCH_SIZE) * EPOCHS
        print(f"FLOPs per Sample: {flops_per_sample:,}")
        print(f"Total FLOPs: {total_flops:,}")
        
        # Память
        memory_weights = (self.w_input.nbytes + self.w_hidden.nbytes) / 1024 / 1024
        memory_activations = (BATCH_SIZE * HIDDEN_SIZE * TIME_STEPS * 4) / 1024 / 1024  # float32
        print(f"Memory (Weights): {memory_weights:.2f} MB")
        print(f"Memory (Activations): {memory_activations:.2f} MB")
        
        # Confusion Matrix
        _, all_preds, all_labels = self.evaluate(self.scaler.transform(X_test), y_test)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Графики
        plt.figure(figsize=(20, 15))
        
        plt.subplot(3, 3, 1)
        plt.plot(self.history['accuracy'], label='Train')
        plt.plot(self.history['val_accuracy'], label='Validation')
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.subplot(3, 3, 2)
        plt.plot(self.history['loss'])
        plt.title("Loss")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 3)
        plt.plot(self.history['spike_rate'])
        plt.title("Spike Rate (Spikes/neuron)")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 4)
        plt.plot(self.history['time_per_epoch'])
        plt.title("Time per Epoch (s)")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 5)
        plt.plot(self.history['energy_per_epoch'])
        plt.title("Energy per Epoch (Wh)")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 6)
        plt.plot([self.history['accuracy'][i] - self.history['val_accuracy'][i] for i in range(EPOCHS)])
        plt.title("Train-Val Accuracy Gap")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 7)
        plt.plot(np.cumsum(self.history['energy_per_epoch']))
        plt.title("Cumulative Energy (Wh)")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 8)
        plt.plot([lr * self.history['spike_rate'][i] for i, lr in enumerate([INIT_LR] * EPOCHS)])
        plt.title("Learning Rate * Spike Rate")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 9)
        plt.plot([self.history['loss'][i] / (self.history['spike_rate'][i] + 1e-5) for i in range(EPOCHS)])
        plt.title("Loss per Spike")
        plt.xlabel("Epoch")
        
        plt.tight_layout()
        plt.savefig('training_analysis.png')
        plt.close()

# Проверка GPU
print("Initializing GPU...")
cp.cuda.Device(0).use()
print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# Загрузка данных
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int32)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Обучение и тестирование
snn = IzhikevichSNN()
print("\nTraining SNN...")
snn.train(X_train, y_train, X_val, y_val)

test_acc, _, _ = snn.evaluate(snn.scaler.transform(X_test), y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Аналитика после тренировки
snn.post_training_analysis(X_train)