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
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10
TIME_STEPS = 20
EPOCHS = 30
BATCH_SIZE = 128
TDP = 160  # Вт, TDP RTX 4060 Ti
STDP_LR = 0.01
HEBBIAN_LR = 0.005
STDP_TAU_PLUS = 15.0
STDP_TAU_MINUS = 15.0
STDP_A_PLUS = 0.02
STDP_A_MINUS = 0.02
WEIGHT_DECAY = 1e-4

class IzhikevichSNN:
    def __init__(self):
        self.w_input = cp.random.normal(0, 1.0, (INPUT_SIZE, HIDDEN_SIZE)) * 0.3
        self.w_hidden = cp.random.normal(0, 1.0, (HIDDEN_SIZE, OUTPUT_SIZE)) * 0.3
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.neuron_params = {
            'a': 0.02,
            'b': 0.2,
            'c': -65.0,
            'd': 8.0,
            'threshold': 20.0
        }
        
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'spike_rate': [],
            'time_per_epoch': [],
            'energy_per_epoch': [],
            'mean_outputs': [],
            'mean_outputs_per_class': [],
            'mean_dw_stdp': [],
            'mean_dw_hebbian': []
        }

        with open('training_metrics_stdp.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Acc', 'Val Acc', 'Spikes/neuron', 'Time (s)', 'Energy (Wh)', 
                           'Mean I_h', 'Input Spikes', 'Mean Outputs', 'Var w_input', 'Var w_hidden', 
                           'Mean Outputs per Class', 'Mean |dw| STDP', 'Mean |dw| Hebbian'])

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

    def encode_input(self, x):
        x = cp.asarray(x)
        spike_train = cp.random.rand(TIME_STEPS, x.shape[0], INPUT_SIZE) < x
        return spike_train.astype(cp.float32)

    def stdp_update(self, pre_spikes, post_spikes, weights):
        dw = cp.zeros_like(weights)
        for t in range(TIME_STEPS):
            pre_t = pre_spikes[t]
            post_t = post_spikes[t]
            for s in range(TIME_STEPS):
                delta_t = t - s
                if delta_t == 0:
                    continue
                pre_s = pre_spikes[s]
                post_s = post_spikes[s]
                if delta_t > 0:
                    ltp = STDP_A_PLUS * cp.exp(-float(delta_t) / STDP_TAU_PLUS)
                    dw += ltp * cp.mean(pre_s[:, :, None] * post_t[:, None, :], axis=0)
                if delta_t < 0:
                    ltd = STDP_A_MINUS * cp.exp(float(delta_t) / STDP_TAU_MINUS)
                    dw -= ltd * cp.mean(pre_t[:, :, None] * post_s[:, None, :], axis=0)
        weights += STDP_LR * dw / TIME_STEPS
        weights = cp.clip(weights, -1.0, 1.0)
        norm = cp.linalg.norm(weights, axis=0, keepdims=True)
        weights = cp.where(norm > 2.0, weights / cp.maximum(norm, 1e-8) * 2.0, weights)
        return weights, cp.mean(cp.abs(dw)).get()

    def hebbian_update(self, spikes_h, outputs, targets, weights):
        outputs = outputs - cp.mean(outputs, axis=1, keepdims=True)
        error = targets - outputs
        dw = HEBBIAN_LR * cp.dot(spikes_h.T, error) / spikes_h.shape[0]
        dw -= WEIGHT_DECAY * weights
        weights += dw
        weights = cp.clip(weights, -1.0, 1.0)
        return weights, cp.mean(cp.abs(dw)).get()

    def forward(self, x, y=None, train_mode=False):
        batch_size = x.shape[0]
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
            I_h = cp.dot(spike_train[t], self.w_input) * 50
            mean_i_h += cp.mean(cp.abs(I_h)).get() / TIME_STEPS
            v_h, u_h, spikes_h = self.izhikevich_update(v_h, u_h, I_h)
            spikes_h_sum += spikes_h
            spike_count += cp.sum(spikes_h)
            post_spikes_h[t] = spikes_h
            
            I_o = cp.dot(spikes_h, self.w_hidden) * 50
            v_o, u_o, spikes_o = self.izhikevich_update(v_o, u_o, I_o)
            output += spikes_o
        
        outputs = output / TIME_STEPS
        mean_outputs = cp.mean(outputs).get()
        outputs_per_class = None
        if y is not None:
            outputs_per_class = []
            for cls in range(OUTPUT_SIZE):
                mask = y == cls
                if cp.sum(mask) > 0:
                    outputs_per_class.append(cp.mean(outputs[mask]).get())
                else:
                    outputs_per_class.append(0.0)
            outputs_per_class = np.array(outputs_per_class)
        
        if train_mode:
            return outputs, spike_count, spikes_h_sum / TIME_STEPS, spike_train, post_spikes_h, mean_i_h, input_spikes, mean_outputs, outputs_per_class
        return outputs, spike_count, spikes_h_sum / TIME_STEPS

    def train(self, X_train, y_train, X_val, y_val):
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
            total_outputs_per_class = np.zeros(OUTPUT_SIZE)
            total_dw_stdp = 0.0
            total_dw_hebbian = 0.0
            batch_count = 0
            
            indices = cp.random.permutation(len(X_train))
            X_shuffled = X_train_gpu[indices]
            y_shuffled = y_train_gpu[indices]
            
            for i in tqdm(range(0, len(X_shuffled), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}"):
                batch_X = X_shuffled[i:i+BATCH_SIZE]
                batch_y = y_shuffled[i:i+BATCH_SIZE]
                
                outputs, batch_spikes, spikes_h_batch, pre_spikes, post_spikes_h, mean_i_h, input_spikes, mean_outputs, outputs_per_class = self.forward(batch_X, batch_y, train_mode=True)
                total_spikes += batch_spikes
                total_i_h += mean_i_h
                total_input_spikes += input_spikes
                total_mean_outputs += mean_outputs
                if outputs_per_class is not None:
                    total_outputs_per_class += outputs_per_class
                batch_count += 1
                
                self.w_input, dw_stdp = self.stdp_update(pre_spikes, post_spikes_h, self.w_input)
                total_dw_stdp += dw_stdp
                
                targets = cp.eye(OUTPUT_SIZE)[batch_y]
                self.w_hidden, dw_hebbian = self.hebbian_update(spikes_h_batch, outputs, targets, self.w_hidden)
                total_dw_hebbian += dw_hebbian
                
                preds = cp.argmax(outputs, axis=1)
                correct += cp.sum(preds == batch_y).get()
            
            acc = correct / len(X_shuffled)
            spike_rate = total_spikes / (len(X_shuffled) * HIDDEN_SIZE * TIME_STEPS)
            val_acc = self.evaluate(X_val_gpu, y_val)[0]
            epoch_time = time.time() - start_time
            energy_epoch = TDP * (epoch_time / 3600)
            mean_i_h = total_i_h / batch_count
            mean_input_spikes = total_input_spikes / batch_count
            mean_outputs = total_mean_outputs / batch_count
            mean_outputs_per_class = total_outputs_per_class / batch_count
            mean_dw_stdp = total_dw_stdp / batch_count
            mean_dw_hebbian = total_dw_hebbian / batch_count
            var_w_input = cp.var(self.w_input).get()
            var_w_hidden = cp.var(self.w_hidden).get()
            
            self.history['accuracy'].append(acc)
            self.history['spike_rate'].append(spike_rate.get() if isinstance(spike_rate, cp.ndarray) else spike_rate)
            self.history['val_accuracy'].append(val_acc)
            self.history['time_per_epoch'].append(epoch_time)
            self.history['energy_per_epoch'].append(energy_epoch)
            self.history['mean_outputs'].append(mean_outputs)
            self.history['mean_outputs_per_class'].append(mean_outputs_per_class.tolist())
            self.history['mean_dw_stdp'].append(mean_dw_stdp)
            self.history['mean_dw_hebbian'].append(mean_dw_hebbian)
            
            log_message = (f"Epoch {epoch+1} | Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | "
                          f"Spikes: {spike_rate:.2f}/neuron | Time: {epoch_time:.1f}s | Energy: {energy_epoch:.2f} Wh | "
                          f"Mean I_h: {mean_i_h:.2f} | Input Spikes: {mean_input_spikes:.4f} | Mean Outputs: {mean_outputs:.4f} | "
                          f"Var w_input: {var_w_input:.4f} | Var w_hidden: {var_w_hidden:.4f} | "
                          f"Mean Outputs per Class: {mean_outputs_per_class.tolist()} | "
                          f"Mean |dw| STDP: {mean_dw_stdp:.6f} | Mean |dw| Hebbian: {mean_dw_hebbian:.6f}")
            print(log_message)
            
            with open('training_metrics_stdp.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, acc, val_acc, spike_rate, epoch_time, energy_epoch, mean_i_h, 
                               mean_input_spikes, mean_outputs, var_w_input, var_w_hidden, 
                               mean_outputs_per_class.tolist(), mean_dw_stdp, mean_dw_hebbian])

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
            all_preds.extend(preds.get().tolist())
            all_labels.extend(batch_y.tolist())
        accuracy = correct / total
        return accuracy, all_preds, all_labels

    def post_training_analysis(self, X_train, X_test, y_test):
        total_time = sum(self.history['time_per_epoch'])
        total_energy = sum(self.history['energy_per_epoch'])
        energy_per_sample = total_energy / len(X_train)
        print(f"\nPost-Training Analysis:")
        print(f"Total Training Time: {total_time:.1f} s")
        print(f"Total Energy Used: {total_energy:.2f} Wh")
        print(f"Energy per Sample: {energy_per_sample:.6f} Wh/sample")
        
        flops_per_sample = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE) * TIME_STEPS
        total_flops = flops_per_sample * (len(X_train) // BATCH_SIZE) * EPOCHS
        print(f"FLOPs per Sample: {flops_per_sample:,}")
        print(f"Total FLOPs: {total_flops:,}")
        
        memory_weights = (self.w_input.nbytes + self.w_hidden.nbytes) / 1024 / 1024
        memory_activations = (BATCH_SIZE * HIDDEN_SIZE * TIME_STEPS * 4) / 1024 / 1024
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
        plt.plot(self.history['mean_outputs'])
        plt.title("Mean Outputs")
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
        plt.plot([STDP_LR * self.history['spike_rate'][i] for i in range(EPOCHS)])
        plt.title("STDP Learning Rate * Spike Rate")
        plt.xlabel("Epoch")
        
        plt.subplot(3, 3, 9)
        plt.plot([self.history['mean_outputs'][i] / (self.history['spike_rate'][i] + 1e-5) for i in range(EPOCHS)])
        plt.title("Mean Outputs per Spike")
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
print("\nTraining SNN with STDP...")
snn.train(X_train, y_train, X_val, y_val)

test_acc = snn.evaluate(snn.scaler.transform(X_test), y_test)[0]
print(f"Test Accuracy: {test_acc:.4f}")

# Аналитика после тренировки
snn.post_training_analysis(X_train, X_test, y_test)