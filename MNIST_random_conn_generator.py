import numpy as np
import os
import argparse

def sparsen_matrix(base_matrix, p_conn):
    """Создаёт разреженную матрицу с заданной вероятностью подключения."""
    num_weights = int(base_matrix.size * p_conn)
    weight_matrix = np.zeros_like(base_matrix, dtype=np.float32)
    weight_list = []
    indices = np.random.choice(base_matrix.size, num_weights, replace=False)
    rows, cols = np.unravel_index(indices, base_matrix.shape)
    for i, j in zip(rows, cols):
        weight_matrix[i, j] = base_matrix[i, j]
        weight_list.append((i, j, float(base_matrix[i, j])))
    return weight_matrix, weight_list

def generate_weights(args):
    """Генерирует и сохраняет веса соединений."""
    n_input = 784
    n_e = args.n_e
    n_i = n_e
    data_path = args.data_path
    os.makedirs(data_path, exist_ok=True)

    weights = {
        'ee_input': 0.3,  # XeAe
        'ei_input': 0.2,  # XeAi
        'ei': 10.4,       # AeAi
        'ie': 17.0        # AiAe
    }
    p_conn = {
        'ee_input': 1.0,
        'ei_input': 0.1,
        'ei': 0.0025,
        'ie': 0.9
    }

    print("Generating XeAe weights")
    weight_matrix = np.random.random((n_input, n_e)).astype(np.float32) * weights['ee_input'] + 0.01
    if p_conn['ee_input'] < 1.0:
        weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ee_input'])
    else:
        weight_list = [(i, j, float(weight_matrix[i, j])) for i in range(n_input) for j in range(n_e)]
    np.save(os.path.join(data_path, 'XeAe.npy'), np.array(weight_list, dtype=object))

    print("Generating XeAi weights")
    weight_matrix = np.random.random((n_input, n_i)).astype(np.float32) * weights['ei_input']
    weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ei_input'])
    np.save(os.path.join(data_path, 'XeAi.npy'), np.array(weight_list, dtype=object))

    print("Generating AeAi weights")
    weight_list = [(i, i, weights['ei']) for i in range(n_e)]
    np.save(os.path.join(data_path, 'AeAi.npy'), np.array(weight_list, dtype=object))

    print("Generating AiAe weights")
    weight_matrix = np.ones((n_i, n_e), dtype=np.float32) * weights['ie']
    np.fill_diagonal(weight_matrix, 0)
    weight_list = [(i, j, float(weight_matrix[i, j])) for i in range(n_i) for j in range(n_e) if weight_matrix[i, j] != 0]
    np.save(os.path.join(data_path, 'AiAe.npy'), np.array(weight_list, dtype=object))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weights for SNN")
    parser.add_argument('--n_e', type=int, default=400, help="Number of excitatory neurons")
    parser.add_argument('--data_path', type=str, default='./random/', help="Path to save weights")
    args = parser.parse_args()
    generate_weights(args)