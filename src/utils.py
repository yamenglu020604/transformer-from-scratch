import matplotlib.pyplot as plt
import os

def plot_and_save_curves(train_losses, val_losses, results_dir):
    """Plots training and validation loss curves and saves the plot."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(results_dir, 'loss_curves.png')
    plt.savefig(save_path)
    print(f"Loss curves saved to {save_path}")
    plt.close()