# torch_cifar_classifier
Practice project

# Torch CIFAR-10 Classifier

A practice project demonstrating how to build, train, and evaluate a convolutional neural network (CNN) on the CIFAR-10 dataset using PyTorch. This repository includes:

- Data loading and preprocessing with `torchvision`
- Definition of a custom CNN (`Net` / `MyNetwork`)
- Training loop with Adam optimizer and Cross-Entropy Loss
- Model evaluation on test data
- Device-agnostic code supporting CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU
- Utility functions for visualization and performance monitoring
- Safe multiprocessing setup for DataLoader workers


## Features

- Clean, modular PyTorch code without high-level abstractions
- Device selection logic for CUDA, Apple MPS, or CPU
- Multiprocessing-safe DataLoader with configurable workers
- Adam optimizer and Cross-Entropy loss for efficient training
- TorchScript export for model deployment
- Examples of best practices for debugging and profiling

## Prerequisites

- Python 3.8 or later  
- PyTorch 1.11 or later  
- Torchvision  
- matplotlib  
- numpy  

(Optional) For GPU support on Apple Silicon, ensure you have a PyTorch build with MPS support.

## Installation

You can install all Python dependencies using the provided `requirements.txt`.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/torch_cifar_classifier.git
   cd torch_cifar_classifier
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


## Usage

Run training and evaluation with:
```bash
python train.py
```
The script will:
1. Set up device (CUDA, MPS, or CPU)
2. Load and preprocess CIFAR-10 dataset
3. Initialize the model, loss, and Adam optimizer
4. Train for a specified number of epochs
5. Evaluate on the test set
6. Display sample images and training statistics

### Configuration

Modify hyperparameters at the top of `train.py`:
```python
batch_size = 16
num_epochs = 2
learning_rate = 1e-3
num_workers = 2
```

## Training

Inside `train.py`, the `train_network` function implements:
- Forward pass
- Loss computation (`CrossEntropyLoss`)
- Backward pass (`loss.backward()`)
- Parameter updates (`optimizer.step()`)

Progress is printed every N mini-batches. To visualize training speed, you can enable `tqdm` or add more logging.

## Evaluation

The `test_network` function loops over `testloader` in `torch.no_grad()` mode, computing overall accuracy per class. To modify evaluation metrics, edit this function in `train.py`.

## Custom Dataset

Use `torchvision.datasets.ImageFolder` or write your own `torch.utils.data.Dataset` subclass. See `utils.py` for an example `MyImageDataset` that reads from a CSV file.

## Visualization

- `imshow` in `utils.py` shows a grid of images with labels.
- Use `matplotlib` to plot loss/accuracy curves or leverage TensorBoard / WandB for richer dashboards.

## Troubleshooting

- **Multiprocessing errors**: Wrap your entry-point code in `if __name__ == "__main__":` to safely spawn DataLoader workers.
- **Device Mismatch**: Always call `.to(device)` on both model and data tensors.
- **Hanging process**: Close matplotlib figures (`plt.close()`) or delete DataLoader objects before exiting.

---

requirements.txt content:

torch
torchvision
matplotlib
numpy
tqdm