import matplotlib.pyplot as plt
from config import num_layers, num_heads, d_model, d_inner

def plot_learning_curve(train_accuracies, train_losses, val_accuracies, val_losses, epochs, args):
    # Plotting accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'r-', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'b-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve - Accuracy')
    plt.legend()

    # Plotting losses
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.plot(epochs, val_losses, 'b-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve - Loss')
    plt.legend()

    # Adjusting layout and saving the plot
    plt.tight_layout()
    plt.savefig(f'lr_curves/learning_curve_{args.model}_{args.modality}_{args.level}_{num_layers}_{num_heads}_{d_model}_{d_inner}__{args.batch_size}_{args.loss}.png')