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
        if i >= base_matrix.shape[0] or j >= base_matrix.shape[1]:
            print(f"Invalid index detected: i={i}, j={j}, matrix shape={base_matrix.shape}")
            continue
        weight_matrix[i, j] = base_matrix[i, j]
        weight_list.append((int(i), int(j), float(base_matrix[i, j])))
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

    # XeAe
    print(f"Generating XeAe weights (shape: ({n_input}, {n_e}))")
    weight_matrix = np.random.random((n_input, n_e)).astype(np.float32) * weights['ee_input'] + 0.01
    if p_conn['ee_input'] < 1.0:
        weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ee_input'])
    else:
        weight_list = []
        for i in range(n_input):
            for j in range(n_e):
                weight_list.append((int(i), int(j), float(weight_matrix[i, j])))
    print(f"XeAe weight_list length: {len(weight_list)}, sample: {weight_list[:5]}")
    # Проверка индексов
    invalid_indices = [(i, j) for i, j, w in weight_list if i >= n_input or j >= n_e]
    if invalid_indices:
        raise ValueError(f"Invalid indices in XeAe: {invalid_indices[:10]}")
    structured_array = np.array(weight_list, dtype=[('i', np.int32), ('j', np.int32), ('w', np.float32)])
    np.save(os.path.join(data_path, 'XeAe.npy'), structured_array)

    # XeAi
    print(f"Generating XeAi weights (shape: ({n_input}, {n_i}))")
    weight_matrix = np.random.random((n_input, n_i)).astype(np.float32) * weights['ei_input']
    weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ei_input'])
    print(f"XeAi weight_list length: {len(weight_list)}, sample: {weight_list[:5]}")
    invalid_indices = [(i, j) for i, j, w in weight_list if i >= n_input or j >= n_i]
    if invalid_indices:
        raise ValueError(f"Invalid indices in XeAi: {invalid_indices[:10]}")
    structured_array = np.array(weight_list, dtype=[('i', np.int32), ('j', np.int32), ('w', np.float32)])
    np.save(os.path.join(data_path, 'XeAi.npy'), structured_array)

    # AeAi
    print(f"Generating AeAi weights (shape: ({n_e}, {n_e}))")
    weight_list = [(int(i), int(i), weights['ei']) for i in range(n_e)]
    print(f"AeAi weight_list length: {len(weight_list)}, sample: {weight_list[:5]}")
    invalid_indices = [(i, j) for i, j, w in weight_list if i >= n_e or j >= n_e]
    if invalid_indices:
        raise ValueError(f"Invalid indices in AeAi: {invalid_indices[:10]}")
    structured_array = np.array(weight_list, dtype=[('i', np.int32), ('j', np.int32), ('w', np.float32)])
    np.save(os.path.join(data_path, 'AeAi.npy'), structured_array)

    # AiAe
    print(f"Generating AiAe weights (shape: ({n_i}, {n_e}))")
    weight_matrix = np.ones((n_i, n_e), dtype=np.float32) * weights['ie']
    np.fill_diagonal(weight_matrix, 0)
    weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ie'])
    print(f"AiAe weight_list length: {len(weight_list)}, sample: {weight_list[:5]}")
    invalid_indices = [(i, j) for i, j, w in weight_list if i >= n_i or j >= n_e]
    if invalid_indices:
        raise ValueError(f"Invalid indices in AiAe: {invalid_indices[:10]}")
    structured_array = np.array(weight_list, dtype=[('i', np.int32), ('j', np.int32), ('w', np.float32)])
    np.save(os.path.join(data_path, 'AiAe.npy'), structured_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weights for SNN")
    parser.add_argument('--n_e', type=int, default=200, help="Number of excitatory neurons")
    parser.add_argument('--data_path', type=str, default='./random/', help="Path to save weights")
    args = parser.parse_args()
    generate_weights(args)
