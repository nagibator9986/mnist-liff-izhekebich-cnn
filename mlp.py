import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import matplotlib.pyplot as plt
import logging
import csv

# Настройка логирования
logging.basicConfig(filename='honest_cnn_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Параметры
EPOCHS = 30
BATCH_SIZE = 128
INPUT_SHAPE = (28, 28, 1)
INIT_LR = 0.02  # Как у Ижикевича
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

# Модель CNN
model = models.Sequential([
    layers.Flatten(input_shape=INPUT_SHAPE),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Компиляция
optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Подсчет FLOPs (приблизительно)
def get_flops(model):
    flops = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            flops += layer.input_shape[-1] * layer.output_shape[-1] * 2  # Умножение + сложение
    return flops

flops_per_sample = get_flops(model)
total_flops = flops_per_sample * (len(X_train) // BATCH_SIZE) * EPOCHS

# Обучение с аналитикой
logging.info("Training CNN...")
print("\nTraining CNN...")
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'time_per_epoch': [], 'energy_per_epoch': []}

for epoch in range(EPOCHS):
    start_time = time.time()
    hist = model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, 
                     validation_data=(X_val, y_val), verbose=1)
    epoch_time = time.time() - start_time
    energy_epoch = TDP * (epoch_time / 3600)  # Вт·ч
    
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
    
    with open('honest_cnn_metrics.csv', 'a' if epoch > 0 else 'w', newline='') as f:
        writer = csv.writer(f)
        if epoch == 0:
            writer.writerow(['Epoch', 'Train Acc', 'Val Acc', 'Loss', 'Val Loss', 'Time (s)', 'Energy (Wh)'])
        writer.writerow([epoch+1, hist.history['accuracy'][0], hist.history['val_accuracy'][0], 
                         hist.history['loss'][0], hist.history['val_loss'][0], epoch_time, energy_epoch])

# Оценка на тесте и сбор предсказаний
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Пост-тренировочная аналитика
total_time = sum(history['time_per_epoch'])
total_energy = sum(history['energy_per_epoch'])
energy_per_sample = total_energy / len(X_train)

print(f"\nPost-Training Analysis:")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Total Training Time: {total_time:.1f} s")
print(f"Total Energy Used: {total_energy:.2f} Wh")
print(f"Energy per Sample: {energy_per_sample:.6f} Wh/sample")
print(f"FLOPs per Sample: {flops_per_sample:,}")
print(f"Total FLOPs: {total_flops:,}")

# Эквивалент спайков
avg_activation = 0.5  # Предположение: 50% нейронов активны (ReLU)
spike_equiv = avg_activation * 512 * len(X_train) / EPOCHS / BATCH_SIZE
print(f"Equivalent Spikes/neuron: {spike_equiv:.2f}")

# Память
memory_weights = sum([w.nbytes for w in model.get_weights()]) / 1024 / 1024
memory_activations = (BATCH_SIZE * 512 * 4) / 1024 / 1024  # float32
print(f"Memory (Weights): {memory_weights:.2f} MB")
print(f"Memory (Activations): {memory_activations:.2f} MB")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
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
plt.plot([spike_equiv] * EPOCHS)
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
plt.plot([history['accuracy'][i] - history['val_accuracy'][i] for i in range(EPOCHS)])
plt.title("Train-Val Accuracy Gap")
plt.xlabel("Epoch")

plt.subplot(3, 3, 7)
plt.plot(np.cumsum(history['energy_per_epoch']))
plt.title("Cumulative Energy (Wh)")
plt.xlabel("Epoch")

plt.subplot(3, 3, 8)
plt.plot([INIT_LR * spike_equiv for _ in range(EPOCHS)])
plt.title("Learning Rate * Spike Rate")
plt.xlabel("Epoch")

plt.subplot(3, 3, 9)
plt.plot([history['loss'][i] / (spike_equiv + 1e-5) for i in range(EPOCHS)])
plt.title("Loss per Spike")
plt.xlabel("Epoch")

plt.tight_layout()
plt.savefig('training_analysis.png')
plt.close()

# Почему этот код максимально честный?
# 1. Одинаковое количество параметров
# Ижикевич: Веса: ( 784 \times 512 + 512 \times 10 = 401,920 + 5,120 = 407,040 ). CNN: Веса: ( 784 \times 512 + 512 \times 10 = 401,920 + 5,120 = 407,040 ). Почему это важно: Количество параметров определяет емкость модели. Если бы у CNN было больше параметров (например, из-за сложных свёрточных слоев), она могла бы иметь несправедливое преимущество. Здесь обе модели имеют идентичное число весов, что делает их сравнение честным с точки зрения сложности.
# 2. Схожая структура сети
# Ижикевич: Два слоя — входной (784) → скрытый (512) → выходной (10). CNN: Два слоя — Flatten (784) → Dense (512) → Dense (10). Почему это важно: Я убрал свёрточные слои и пулинг, которые могли бы дать CNN преимущество за счет извлечения пространственных признаков. Вместо этого CNN работает как полносвязная сеть, аналогично Ижикевичу, где входные пиксели напрямую передаются в скрытый слой.
# 3. Одинаковые гиперпараметры
# Эпохи: Обе модели обучаются 30 эпох. Размер батча: ( BATCH_SIZE = 128 ) в обоих случаях. Оптимизатор: Adam с ( INITLR = 0.02 ), ( \beta1 = 0.9 ), ( \beta_2 = 0.999 ) — те же параметры, что у Ижикевича в финальной версии. Dropout: 30% в обеих моделях для контроля переобучения. Почему это важно: Одинаковые условия обучения исключают влияние внешних факторов (например, более быстрая сходимость из-за меньшего шага обучения или большего числа эпох).
# 4. Те же данные
# Ижикевич: MNIST, 784 пикселя, нормализованные в [0, 1]. CNN: MNIST, 28x28x1, нормализованные в [0, 1]. Почему это важно: Обе модели получают идентичный входной сигнал (пиксели MNIST), хотя Ижикевич преобразует их в спайки через rate coding, а CNN использует их напрямую. Это честно, так как входные данные не модифицированы в пользу одной из моделей.
# 5. Отсутствие архитектурных "читов"
# CNN: Нет свёрточных слоев, пулинга или дополнительных скрытых слоев, которые могли бы повысить точность за счет пространственной обработки или глубины. Почему это важно: Типичная CNN на MNIST (с Conv2D и MaxPooling) легко достигает 98–99%, но это было бы нечестно, так как Ижикевич ограничен двумя слоями и спайковой динамикой. Я убрал эти преимущества, сделав CNN максимально близкой к Ижикевичу по структуре.
# 6. Сравнимая вычислительная среда
# Ижикевич: Использует CuPy на GPU (RTX 4060 Ti). CNN: Использует TensorFlow с GPU-поддержкой (RTX 4060 Ti). Почему это важно: Обе модели работают на одной аппаратной платформе, что исключает влияние разницы в производительности CPU vs GPU.
# 7. Аналитика для справедливого сравнения
# Код включает метрики (точность, время, энергия, FLOPs, память, эквивалент спайков), которые напрямую сравнимы с Ижикевичем. Например: FLOPs: Подсчитаны для каждого слоя, чтобы сравнить вычислительную сложность. Энергия: Оценена через TDP и время, как у Ижикевича. Спайки: Эквивалент спайков для CNN рассчитан как доля активных нейронов (ReLU), чтобы связать с метрикой Ижикевича.
# Возможные возражения и ответы
# "CNN все равно точнее из-за непрерывных активаций":
# Да, это правда — ReLU и softmax дают преимущество в точности (ожидаемо 90–95% vs 76% у Ижикевича). Но это не "нечестность", а фундаментальное различие между SNN и ANN. Я минимизировал архитектурные преимущества CNN, чтобы разница была только в принципах работы (спайки vs непрерывность).
# "Ижикевич использует 50 временных шагов, а CNN нет":
# Это тоже фундаментальная особенность SNN. Я не мог добавить временные шаги в CNN без превращения ее в RNN, что нарушило бы честность. Вместо этого я сделал CNN максимально простой, чтобы ее вычисления были сравнимы с одним проходом Ижикевича.
# "CNN не биоправдоподобна":
# Это не влияет на честность сравнения по эффективности (точность, энергия, время). Биоправдоподобность — преимущество Ижикевича, но ваша задача — сравнение эффективности, а не биологии.
# Почему это "максимально честно"?
# Архитектура: CNN повторяет структуру Ижикевича (784 → 512 → 10) без дополнительных слоев. Параметры: Число весов идентично (407,040). Условия: Те же эпохи, батчи, оптимизатор, данные и dropout. Ограничения: Убраны свёртки и пулинг, чтобы CNN не "читерила" за счет пространственных признаков.
# Если бы я оставил свёрточные слои или увеличил число нейронов, CNN могла бы достичь 98–99%, но это было бы несправедливо, так как Ижикевич ограничен своей спайковой природой и двухслойной архитектурой. Моя цель — показать, как обе модели работают в максимально схожих условиях, чтобы вы могли оценить их эффективность (точность, энергию, время) без перекоса.