import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import gzip
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_labeled_data(path, bTrain=True):
    """Загружает данные MNIST."""
    with gzip.open(os.path.join(path, 'training.pkl.gz' if bTrain else 'testing.pkl.gz'), 'rb') as f:
        data = pickle.load(f)
    return {'X': data[0].reshape((-1, 784)).astype(np.float32), 'y': data[1].astype(np.int32)}

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

def plot_confusion_matrix(true_labels, pred_labels, save_path):
    """Строит матрицу ошибок."""
    cm = confusion_matrix(true_labels, pred_labels, labels=range(10))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def plot_error_examples(data, true_labels, pred_labels, save_path):
    """Визуализирует ошибочные предсказания."""
    errors = np.where(true_labels != pred_labels)[0]
    if len(errors) == 0:
        print("No errors found")
        return
    n_examples = min(16, len(errors))
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(errors[:n_examples]):
        plt.subplot(4, 4, i+1)
        plt.imshow(data['X'][idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_labels[idx]}, Pred: {pred_labels[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'error_examples.png'))
    plt.close()

def main(args):
    n_e = args.n_e if args.n_e2 == 0 else args.n_e2  # Используем второй слой, если есть
    save_path = args.save_path
    num_examples = args.num_examples

    # Загрузка данных
    print("Loading MNIST")
    training = get_labeled_data(args.mnist_path, bTrain=True)
    testing = get_labeled_data(args.mnist_path, bTrain=False)

    # Загрузка результатов
    print("Loading results")
    train_result = np.load(os.path.join(save_path, 'activity', 'result_monitor_final.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(save_path, 'activity', 'input_numbers_final.npy'), allow_pickle=True)
    test_result = np.load(os.path.join(save_path, 'activity', 'result_monitor_final.npy'), allow_pickle=True)
    test_labels = np.load(os.path.join(save_path, 'activity', 'input_numbers_final.npy'), allow_pickle=True)

    # Назначение классов
    print("Assigning classes")
    assignments = get_new_assignments(train_result, train_labels, n_e)

    # Оценка
    print("Evaluating accuracy")
    test_results = np.zeros((10, num_examples), dtype=np.int32)
    predictions = np.zeros(num_examples, dtype=np.int32)
    for i in range(num_examples):
        test_results[:, i] = get_recognized_number_ranking(assignments, test_result[i])
        predictions[i] = test_results[0, i]
    correct = np.sum(predictions == test_labels)
    accuracy = correct / num_examples * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Анализ ошибок
    print("Analyzing errors")
    plot_confusion_matrix(test_labels, predictions, save_path)
    plot_error_examples(testing, test_labels, predictions, save_path)

    # Визуализация назначений
    plt.hist(assignments, bins=10, range=(-0.5, 9.5))
    plt.title('Neuron Assignments')
    plt.xlabel('Class')
    plt.ylabel('Number of Neurons')
    plt.savefig(os.path.join(save_path, 'assignments.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SNN on MNIST")
    parser.add_argument('--n_e', type=int, default=400, help="Number of excitatory neurons in first layer")
    parser.add_argument('--n_e2', type=int, default=200, help="Number of excitatory neurons in second layer")
    parser.add_argument('--num_examples', type=int, default=10000, help="Number of test examples")
    parser.add_argument('--mnist_path', type=str, default='./mnist/', help="Path to MNIST")
    parser.add_argument('--save_path', type=str, default='./results/', help="Path to saved results")
    args = parser.parse_args()
    main(args)