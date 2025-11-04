# Transformer from Scratch

## 1. Project Overview

This project is a mid-term assignment for the "Fundamentals and Applications of Large Models" course. It involves implementing a complete **Transformer** architecture from scratch using PyTorch, as detailed in the paper "Attention Is All You Need" by Vaswani et al.


## 2. Project Structure

```
assignment-transformer/
│
├── src/                  # Source code directory
│   ├── model.py          # Core Transformer model architecture
│   ├── dataset.py        # Data loading, tokenization, and Dataset creation
│   ├── train.py          # Main script for training and evaluation
│   └── utils.py          # Helper functions (e.g., for plotting)
│
├── scripts/
│   └── run.sh            # One-click script to reproduce the experiment
│
├── configs/
│   └── base_config.yaml  # All hyperparameters and settings
│
├── results/              # Output directory for training curves and tables
│
├── requirements.txt      # Project dependencies
│   
└── README.md             # This documentation

```

## 3. Setup and Installation

**Step 1: Clone the repository**
```bash
git clone <your-repo-link>
cd assignment-transformer
```

**Step 2: Create a Conda environment and install dependencies**

```bash
# Create and activate a conda environment (Python 3.10 is recommended)
conda create -n transformer python=3.10 -y
conda activate transformer

# Install dependencies
pip install -r requirements.txt
```

## 4. How to Run the Experiment

This project is designed for easy reproducibility. Simply run the `run.sh` script from the root directory of the project.

**Exact command to reproduce the experiment:**
```bash
bash scripts/run.sh
```

**What this script does:**
1. Sets a global random seed for reproducibility.
2. Creates the `checkpoints`, `results`, and `tokenizer` directories if they don't exist.
3. Executes the main training script `src/train.py` using the hyperparameters defined in `configs/base_config.yaml`.

The script will first download the IWSLT2017 dataset and train BPE tokenizers for English and German (this only happens on the first run). Then, it will start the training process.

## 5. Expected Results

After the script finishes, you can expect the following outputs:

- **Trained Model**: The best model checkpoint (based on validation loss) will be saved at `checkpoints/best_model.pt`.
- **Training Curves**: A plot showing the training and validation loss over epochs will be saved at `results/loss_curves.png`.
- **Tokenizers**: The trained tokenizers will be saved in the `tokenizer/` directory for future use.
- **Console Output**: The script will print the training progress, including loss and learning rate for each step, and a summary of the training/validation loss for each epoch.


## 6. Configuration Details

All hyperparameters can be modified in the `configs/base_config.yaml` file without changing any source code. Key parameters include:
- `d_model`: The main dimension of the model (embeddings, attention outputs).
- `n_heads`: The number of attention heads.
- `n_layers`: The number of layers in both the encoder and decoder.
- `batch_size`: The number of samples per batch.
- `learning_rate`: The base learning rate for the optimizer.
- `epochs`: The total number of training epochs.