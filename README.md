# MNIST Classification with Spiking and Artificial Neural Networks

This repository provides implementations of Spiking Neural Networks (SNNs) and Artificial Neural Networks (ANNs) for classifying the MNIST dataset. The SNNs are based on the Izhikevich neuron model (with gradient-based and STDP learning) and the Brian2 library, while the ANNs include a multilayer perceptron (MLP) and a convolutional neural network (CNN). Additional scripts support evaluation and weight generation for the Brian2 SNN. This README details each file, its functionality, installation requirements, and execution instructions.

---

## File Descriptions

### 1. `izhikevich-gradient.py`
#### Overview
Implements a Spiking Neural Network (SNN) using the Izhikevich neuron model for MNIST classification. The network has a two-layer architecture (784 input → 1024 hidden → 10 output) and is trained with gradient-based learning using a surrogate gradient and the Adam optimizer. Inputs are encoded as spike trains over 50 time steps. Metrics (accuracy, loss, spike rate, energy) are logged, and visualizations are generated.

#### Key Features
- **Architecture**: 784 input → 1024 hidden → 10 output.
- **Learning**: Adam optimizer (`INIT_LR=0.02`, `BETA1=0.9`, `BETA2=0.999`) with surrogate gradient.
- **Input Encoding**: Rate-based spike trains.
- **Metrics**: Training/validation accuracy, loss, spike rate, energy (based on RTX 4060 Ti TDP of 160W), FLOPs, memory.
- **Outputs**: `training_metrics.csv`, `confusion_matrix.png`, `training_analysis.png`.

#### Requirements
```
numpy
cupy
scikit-learn
seaborn
tqdm
matplotlib
```

#### Installation
1. Install dependencies:
   ```bash
   pip install numpy cupy scikit-learn seaborn tqdm matplotlib
   ```
2. Install CuPy for your CUDA version (e.g., CUDA 11.x for RTX 4060 Ti):
   ```bash
   pip install cupy-cuda11x
   ```
   If `pip` fails, use Conda:
   ```bash
   conda install cupy -c conda-forge
   ```
3. Install CUDA Toolkit from NVIDIA's website: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Ensure compatibility with your GPU (e.g., CUDA 11.x or 12.x for RTX 4060 Ti).
4. Verify GPU availability with `nvidia-smi`.

#### Running the Script
1. Save as `izhikevich-gradient.py`.
2. Run:
   ```bash
   python izhikevich-gradient.py
   ```
3. The script will:
   - Download MNIST via `fetch_openml`.
   - Train for 30 epochs (batch size 128).
   - Save metrics and visualizations.
   - Print training progress, test accuracy, and analysis.

#### Notes
- Requires a CUDA-capable GPU. For CPU-only, replace `cp` with `np` and remove GPU code.
- Adjust `TDP` (default: 160W) for your GPU.
- Internet connection needed for MNIST download.

---

### 2. `izhikevichSTDP.py`
#### Overview
Implements an Izhikevich SNN for MNIST classification with STDP for input-to-hidden weights and Hebbian learning for hidden-to-output weights. The network has a smaller hidden layer (512 neurons) and 20 time steps. It logs detailed metrics (weight variances, STDP/Hebbian changes) and generates visualizations.

#### Key Features
- **Architecture**: 784 input → 512 hidden → 10 output.
- **Learning**: STDP (`STDP_LR=0.01`) for input-to-hidden, Hebbian (`HEBBIAN_LR=0.005`) for hidden-to-output.
- **Input Encoding**: Rate-based spike trains.
- **Metrics**: Accuracy, spike rate, energy, mean currents, weight variances, outputs per class, weight changes.
- **Outputs**: `training_metrics_stdp.csv`, `confusion_matrix.png`, `training_analysis.png`.

#### Requirements
```
numpy
cupy
scikit-learn
seaborn
tqdm
matplotlib
```

#### Installation
Same as `izhikevich-gradient.py`:
```bash
pip install numpy cupy scikit-learn seaborn tqdm matplotlib
pip install cupy-cuda11x
```
Or via Conda:
```bash
conda install cupy -c conda-forge
```
Install CUDA Toolkit: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

#### Running the Script
1. Save as `izhikevichSTDP.py`.
2. Run:
   ```bash
   python izhikevichSTDP.py
   ```
3. The script will:
   - Download MNIST.
   - Train for 30 epochs.
   - Save metrics and visualizations.
   - Print results.

#### Notes
- GPU required for CuPy. CPU adaptation possible.
- STDP increases computational load.
- Adjust `TDP` as needed.

---

### 3. `mlp.py`
#### Overview
Implements a Multilayer Perceptron (MLP) using TensorFlow for MNIST classification, designed to match the Izhikevich SNNs in parameter count (407,040) for fair comparison. It uses a two-layer architecture (784 → 512 → 10) with ReLU, dropout, and softmax.

#### Key Features
- **Architecture**: 784 → Dense (512, ReLU) → Dropout (0.3) → Dense (10, softmax).
- **Learning**: Adam (`INIT_LR=0.02`, `beta_1=0.9`, `beta_2=0.999`).
- **Metrics**: Accuracy, loss, energy, FLOPs, memory, equivalent spike rate (50% ReLU activation).
- **Outputs**: `honest_cnn_metrics.csv`, `confusion_matrix.png`, `training_analysis.png`.

#### Requirements
```
numpy
tensorflow
scikit-learn
seaborn
matplotlib
```

#### Installation
```bash
pip install numpy tensorflow scikit-learn seaborn matplotlib
```
Install CUDA Toolkit and cuDNN for GPU support: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

#### Running the Script
1. Save as `mlp.py`.
2. Run:
   ```bash
   python mlp.py
   ```
3. The script will:
   - Download MNIST.
   - Train for 30 epochs.
   - Save metrics, visualizations, and logs (`honest_cnn_log.log`).
   - Print results.

#### Notes
- GPU support requires CUDA and cuDNN.
- Matches SNNs in parameters and hyperparameters.

---

### 4. `cnn.py`
#### Overview
Implements a Convolutional Neural Network (CNN) using TensorFlow for MNIST classification. Features two convolutional layers (32 and 64 filters), batch normalization, max-pooling, and a dense layer (512 neurons), with early stopping.

#### Key Features
- **Architecture**: Conv2D (32, 3x3) → BatchNorm → MaxPool → Conv2D (64, 3x3) → BatchNorm → MaxPool → Dense (512, ReLU) → Dropout (0.3) → Dense (10, softmax).
- **Learning**: Adam (`INIT_LR=0.001`, `beta_1=0.9`, `beta_2=0.999`), early stopping (patience=5).
- **Metrics**: Accuracy, loss, energy, FLOPs, memory, equivalent spike rate.
- **Outputs**: `cnn_metrics_fixed.csv`, `confusion_matrix_fixed.png`, `cnn_training_analysis_fixed.png`.

#### Requirements
```
numpy
tensorflow
scikit-learn
seaborn
matplotlib
```

#### Installation
```bash
pip install numpy tensorflow scikit-learn seaborn matplotlib
```
Install CUDA Toolkit and cuDNN: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

#### Running the Script
1. Save as `cnn.py`.
2. Run:
   ```bash
   python cnn.py
   ```
3. The script will:
   - Download MNIST.
   - Train for up to 30 epochs with early stopping.
   - Save metrics, visualizations, and logs (`cnn_log_fixed.log`).
   - Print results.

#### Notes
- Lower `INIT_LR=0.001` for stability.
- GPU support enhances performance.

---

### 5. `MNIST_Brian2.py`
#### Overview
Implements an SNN using Brian2 for MNIST classification. Features a Poisson input layer (784 neurons), excitatory and inhibitory layers (n_e and n_i neurons), and an optional second excitatory layer (n_e2). Uses STDP and CPU parallelization.

#### Key Features
- **Architecture**: 784 Poisson → Excitatory (n_e) + Inhibitory (n_i) → Optional Excitatory (n_e2).
- **Learning**: STDP with adaptive thresholds and weight normalization.
- **Input Encoding**: Poisson spike rates.
- **Metrics**: Accuracy, memory usage.
- **Outputs**: Weights (`XeAe.npy`, etc.), activity (`result_monitor_*.npy`), checkpoints, `weights.png`, `accuracy.png`.

#### Requirements
```
numpy
brian2
matplotlib
pickle
gzip
psutil
argparse
```

#### Installation
```bash
pip install numpy brian2 matplotlib psutil cython
```
Note: `pickle`, `gzip`, `argparse` are standard. Install a C++ compiler if needed (e.g., `build-essential` on Ubuntu).

#### Running the Script
1. Save as `MNIST_Brian2.py`.
2. Ensure MNIST data (.pkl.gz) is in `mnist_path` (default: `./mnist/`).
3. Generate weights with `MNIST_random_conn_generator.py` in `data_path` (default: `./`).
4. Run:
   ```bash
   python MNIST_Brian2.py --mnist_path ./mnist/ --data_path ./ --save_path ./results/
   ```
5. Optional arguments:
   - `--n_e`: Excitatory neurons (default: 400).
   - `--n_e2`: Second-layer neurons (default: 200).
   - `--num_examples`: Examples (default: 60000).
   - `--test_mode`: Use pre-trained weights.

#### Notes
- Requires .pkl.gz MNIST data.
- Adjust `num_threads` for CPU parallelization.

---
## Command-line Arguments

The script accepts the following command-line arguments:

| Argument | Type | Default | Description |
|:---|:---|:---|:---|
| `--n_e` | int | 200 | Number of excitatory neurons in the first layer. Higher values can improve model capacity but increase computation time. |
| `--n_e2` | int | 0 | Number of excitatory neurons in the second layer. Set to 0 if no second layer is used. |
| `--example_time` | float | 0.35 | Duration for which each input example is presented (in seconds). Controls how long the network processes each input. |
| `--resting_time` | float | 0.15 | Resting period after each input (in seconds), allowing network activity to decay. |
| `--num_examples` | int | 60000 | Total number of training examples. More examples can lead to better learning but take longer to train. |
| `--update_interval` | int | 30000 | Interval (in examples) at which training statistics are updated and optionally printed/logged. |
| `--weight_update_interval` | int | 100 | Frequency (in iterations) of synaptic weight updates. Smaller values result in more frequent learning updates. |
| `--save_interval` | int | 5000 | Number of examples after which the model state is saved. Useful for checkpointing. |
| `--input_intensity` | float | 2.0 | Scaling factor for the input signals. Higher values result in stronger input currents to neurons. |
| `--weight_sum` | float | 78.0 | Target total synaptic weight per neuron. Helps regulate weight normalization and prevent runaway excitation. |
| `--num_threads` | int | 2 | Number of CPU threads to use for computation. Adjust based on your hardware. |
| `--resume_from` | int | None | If set, resumes training from a specific checkpoint iteration. Useful for continuing interrupted training. |
| `--mnist_path` | str | `./mnist/` | Path to the MNIST dataset directory. |
| `--data_path` | str | `./random/` | Path to initial synaptic weights (random or pretrained). |
| `--save_path` | str | `./results/` | Path where results (models, logs, plots) will be saved. |
| `--test_mode` | flag | - | If set, runs the model in test mode (no training), evaluating performance on test data only. |

---

## Example Usages

Here are some examples of how to use these arguments:

### Train a model with custom settings

```bash
python your_script.py --n_e 400 --example_time 0.5 --num_examples 10000 --input_intensity 1.5 --save_path ./custom_results/
```
- Uses 400 excitatory neurons.
- Each example is shown for 0.5 seconds.
- Trains on 10,000 examples.
- Input intensity is reduced to 1.5.
- Saves outputs to `./custom_results/`.

---

### Resume training from a checkpoint

```bash
python your_script.py --resume_from 30000 --save_path ./continued_training/
```
- Resumes training from iteration 30,000.
- Results are saved in a different folder to avoid overwriting old files.

---

### Run in test mode only

```bash
python your_script.py --test_mode --mnist_path ./data/mnist/ --save_path ./test_results/
```
- No training will occur.
- The script will load the dataset from `./data/mnist/` and save test results to `./test_results/`.

---

## Notes

- **Higher `n_e` and `n_e2`** increase the model's capacity but also require more computational resources.
- **Shorter `example_time` and `resting_time`** make training faster but can harm learning quality if too small.
- **Adjust `num_threads`** based on your machine's CPU cores for better performance.
- **Always make sure** the paths for MNIST data and saving results exist, or they will be created automatically.


### 6. `MNIST_evaluation.py`
#### Overview
Evaluates the Brian2 SNN from `MNIST_Brian2.py`. Loads spike monitors and labels, assigns classes, computes test accuracy, and visualizes results (confusion matrix, error examples, neuron assignments).

#### Key Features
- **Functionality**: Test accuracy, confusion matrix, misclassified examples, neuron assignments.
- **Inputs**: `result_monitor_final.npy`, `input_numbers_final.npy`.
- **Outputs**: `confusion_matrix.png`, `error_examples.png`, `assignments.png`.

#### Requirements
```
numpy
matplotlib
scikit-learn
seaborn
pickle
gzip
argparse
```

#### Installation
```bash
pip install numpy matplotlib scikit-learn seaborn
```

#### Running the Script
1. Save as `MNIST_evaluation.py`.
2. Ensure Brian2 results are in `save_path` (default: `./results/`).
3. Ensure .pkl.gz MNIST data is in `mnist_path` (default: `./mnist/`).
4. Run:
   ```bash
   python MNIST_evaluation.py --mnist_path ./mnist/ --save_path ./results/
   ```
5. Optional arguments:
   - `--n_e`: Excitatory neurons (default: 400).
   - `--n_e2`: Second-layer neurons (default: 200).
   - `--num_examples`: Test examples (default: 10000).

#### Notes
- Requires Brian2 results.
- CPU-based, no GPU needed.

---

### 7. `MNIST_random_conn_generator.py`
#### Overview
Generates initial random weights for the Brian2 SNN. Creates sparse weight matrices for input-to-excitatory (`XeAe`), input-to-inhibitory (`XeAi`), excitatory-to-inhibitory (`AeAi`), and inhibitory-to-excitatory (`AiAe`) connections.

#### Key Features
- **Functionality**: Sparse weight matrices with specified connection probabilities.
- **Outputs**: `XeAe.npy`, `XeAi.npy`, `AeAi.npy`, `AiAe.npy`.

#### Requirements
```
numpy
argparse
```

#### Installation
```bash
pip install numpy
```

#### Running the Script
1. Save as `MNIST_random_conn_generator.py`.
2. Run:
   ```bash
   python MNIST_random_conn_generator.py --data_path ./random/
   ```
3. Optional arguments:
   - `--n_e`: Excitatory neurons (default: 400).
   - `--data_path`: Save path (default: `./random/`).

#### Notes
- Run before `MNIST_Brian2.py`.
- Ensure `data_path` matches `MNIST_Brian2.py`.

---

## General Setup
- **Python Version**: Use Python 3.7+.
- **GPU Support**: For `izhikevich*`, `mlp.py`, `cnn.py`, install CUDA Toolkit and cuDNN from [NVIDIA](https://developer.nvidia.com/cuda-downloads). Verify with `nvidia-smi`.
- **Conda Alternative**: If `pip` fails for CuPy, use:
  ```bash
  conda install cupy -c conda-forge
  ```
- **Data**: `izhikevich*`, `mlp.py`, `cnn.py` download MNIST via `fetch_openml`. `MNIST_Brian2.py` and `MNIST_evaluation.py` require .pkl.gz MNIST data.
- **Permissions**: Ensure write access for saving outputs.
- **Customization**: Adjust hyperparameters and paths as needed.

For issues, contact the repository maintainer or open an issue.
