import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import logging
import csv

# Настройка логирования
logging.basicConfig(filename='cnn_log_fixed.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Параметры
EPOCHS = 30
BATCH_SIZE = 128
INPUT_SHAPE = (28, 28, 1)
INIT_LR = 0.001  # Уменьшено для стабильности
TDP = 160  # Вт, TDP RTX 4060 Ti

# Загрузка данных
logging.info("Loading MNIST...")
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int32)
X = X.values.reshape(-1, 28, 28, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


print("Train class distribution:", np.bincount(y_train))
print("Val class distribution:", np.bincount(y_val))
print("Test class distribution:", np.bincount(y_test))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE, padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

def get_flops(model):
    flops = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            output_shape = layer.output_shape[1:3]
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
            flops += (kernel_size * layer.input_shape[-1] * layer.filters * output_shape[0] * output_shape[1] * 2)
        elif isinstance(layer, tf.keras.layers.Dense):
            flops += layer.input_shape[-1] * layer.output_shape[-1] * 2
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            output_shape = layer.output_shape[1:3]
            pool_size = layer.pool_size[0] * layer.pool_size[1]
            flops += output_shape[0] * output_shape[1] * layer.input_shape[-1] * pool_size
    return flops

flops_per_sample = get_flops(model)
total_flops = flops_per_sample * (len(X_train) // BATCH_SIZE) * EPOCHS

logging.info("Training CNN...")
print("\nTraining CNN...")
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'time_per_epoch': [], 'energy_per_epoch': []}

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

for epoch in range(EPOCHS):
    start_time = time.time()
    hist = model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, 
                     validation_data=(X_val, y_val), verbose=1, callbacks=callbacks)
    epoch_time = time.time() - start_time
    energy_epoch = TDP * (epoch_time / 3600)
    
    history['accuracy'].append(hist.history['accuracy'][0])
    history['val_accuracy'].append(hist.history['val_accuracy'][0])
    history['loss'].append(hist.history['loss'][0])
    history['val_loss'].append(hist.history['val_loss'][0])
    history['time_per_epoch'].append(epoch_time)
    history['energy_per_epoch'].append(energy_epoch)
    
    log_message = (f"Epoch {epoch+1}/{EPOCHS} | Acc: {hist.history['accuracy'][0]:.4f} | "
                   f"Val Acc: {hist.history['val_accuracy'][0]:.4f} | Loss: {hist.history['loss'][0]:.4f} | "
                   f"Time: {epoch_time:.1f}s | Energy: {energy_epoch:.2f} Wh")
    print(log_message)
    logging.info(log_message)
    
    with open('cnn_metrics_fixed.csv', 'a' if epoch > 0 else 'w', newline='') as f:
        writer = csv.writer(f)
        if epoch == 0:
            writer.writerow(['Epoch', 'Train Acc', 'Val Acc', 'Loss', 'Val Loss', 'Time (s)', 'Energy (Wh)'])
        writer.writerow([epoch+1, hist.history['accuracy'][0], hist.history['val_accuracy'][0], 
                         hist.history['loss'][0], hist.history['val_loss'][0], epoch_time, energy_epoch])

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

total_time = sum(history['time_per_epoch'])
total_energy = sum(history['energy_per_epoch'])
energy_per_sample = total_energy / len(X_train)

print(f"\nPost-Training Analysis:")
print(f"Total Training Time: {total_time:.1f} s")
print(f"Total Energy Used: {total_energy:.2f} Wh")
print(f"Energy per Sample: {energy_per_sample:.6f} Wh/sample")
print(f"FLOPs per Sample: {flops_per_sample:,}")
print(f"Total FLOPs: {total_flops:,}")

avg_activation = 0.5
spike_equiv = avg_activation * (32 * 28 * 28 + 64 * 14 * 14 + 512) * len(X_train) / EPOCHS / BATCH_SIZE
print(f"Equivalent Spikes/neuron: {spike_equiv:.2f}")

memory_weights = sum([w.nbytes for w in model.get_weights()]) / 1024 / 1024
memory_activations = (BATCH_SIZE * (32 * 28 * 28 + 64 * 14 * 14 + 512) * 4) / 1024 / 1024
print(f"Memory (Weights): {memory_weights:.2f} MB")
print(f"Memory (Activations): {memory_activations:.2f} MB")

from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_fixed.png')
plt.close()

plt.figure(figsize=(20, 15))

plt.subplot(3, 3, 1)
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(3, 3, 2)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(3, 3, 3)
plt.plot([spike_equiv] * len(history['accuracy']))
plt.title("Equivalent Spike Rate (Spikes/neuron)")
plt.xlabel("Epoch")

plt.subplot(3, 3, 4)
plt.plot(history['time_per_epoch'])
plt.title("Time per Epoch (s)")
plt.xlabel("Epoch")

plt.subplot(3, 3, 5)
plt.plot(history['energy_per_epoch'])
plt.title("Energy per Epoch (Wh)")
plt.xlabel("Epoch")

plt.subplot(3, 3, 6)
plt.plot([history['accuracy'][i] - history['val_accuracy'][i] for i in range(len(history['accuracy']))])
plt.title("Train-Val Accuracy Gap")
plt.xlabel("Epoch")

plt.subplot(3, 3, 7)
plt.plot(np.cumsum(history['energy_per_epoch']))
plt.title("Cumulative Energy (Wh)")
plt.xlabel("Epoch")

plt.subplot(3, 3, 8)
plt.plot([INIT_LR * spike_equiv for _ in range(len(history['accuracy']))])
plt.title("Learning Rate * Spike Rate")
plt.xlabel("Epoch")

plt.subplot(3, 3, 9)
plt.plot([history['loss'][i] / (spike_equiv + 1e-5) for i in range(len(history['accuracy']))])
plt.title("Loss per Spike")
plt.xlabel("Epoch")

plt.tight_layout()
plt.savefig('cnn_training_analysis_fixed.png')
plt.close()