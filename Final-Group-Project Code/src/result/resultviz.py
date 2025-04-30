#tracker

import matplotlib.pyplot as plt
class MetricTracker:
    def __init__(self):
        self.train_losses = []
        self.val_accuracies = []

    def log_train_loss(self, loss):
        self.train_losses.append(loss)

    def log_val_accuracy(self, acc):
        self.val_accuracies.append(acc)

    def plot(self):
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='red', marker='o')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='blue', marker='o')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()