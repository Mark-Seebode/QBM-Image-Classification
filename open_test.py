from typing import Sequence, List
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("out/slurm/acc_per_epoch441init.pkl", "rb") as f:
    acc_list: List[float] = pickle.load(f)

with open("out/slurm/auc_per_epoch441init.pkl", "rb") as f:
    auc_list: List[float] = pickle.load(f)

with open("out/slurm/epoch441init.pkl", "rb") as f:
    loss_list: List[float] = pickle.load(f)

with open("out/slurm/kernel_change_history441init.pkl", "rb") as f:
    kernel_history = pickle.load(f)



# Plot accuracy and AUC per epoch for each run
epochs = list(range(1, len(acc_list) + 1))
plt.figure(figsize=(12, 6))
plt.plot(epochs, acc_list, label='Accuracy', color='blue')
plt.plot(epochs, auc_list, label='AUC', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Accuracy and AUC per Epoch')
plt.legend()
plt.grid()
plt.show()


epochs_loss = list(range(1, len(loss_list) + 1))
# Plot loss per epoch
plt.figure(figsize=(12, 6))
plt.plot(epochs_loss, loss_list, label='Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid()
plt.show()


# Plot kernel changes over epochs as a list of floats
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(kernel_history) + 1), kernel_history, label='Kernel Value', color='green')

plt.xlabel('Epoch')
plt.ylabel('Kernel Value')
plt.title('Kernel Changes over Epochs')
plt.legend()

plt.grid()
plt.show()






