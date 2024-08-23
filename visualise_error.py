import matplotlib.pyplot as plt
import numpy as np

folds = np.arange(5)
models = ["DistilBERT", "DistilBioBERT", "DistilClinicalBERT"]

for model in models:
    plt.figure(figsize=(25, 5))
    plt.suptitle(f"{model} Fine-Tuning with LoRA - Cross Validation")
    # plt.suptitle(f"{model} Fine-Tuning with DoRA - Cross Validation")
    for k in folds:
        # train_error = np.load(f"./models/{model}/LoRA/last_LoRA_3_labels_cv{k}_average_train_losses.npy")
        # val_error = np.load(f"./models/{model}/LoRA/last_LoRA_3_labels_cv{k}_average_val_losses.npy")
        train_error = np.load(f"./models/{model}/DoRA/last_DoRA_3_labels_cv{k}_average_train_losses.npy")
        val_error = np.load(f"./models/{model}/DoRA/last_DoRA_3_labels_cv{k}_average_val_losses.npy")
        plt.subplot(1,5,k+1)
        plt.plot(train_error, "b", label="Avg train loss")
        plt.plot(val_error, "g", label="Avg val loss")

        best_train_loss = np.min(train_error)
        best_train_epoch = np.argmin(train_error)
        plt.plot(best_train_epoch, best_train_loss, "bo", label="Best train loss")
        best_val_loss = np.min(val_error)
        best_val_epoch = np.argmin(val_error)
        plt.plot(best_val_epoch, best_val_loss, "go", label="Best val loss")

        print(f"{model} has the best validation loss at: {best_val_epoch+1}")

        plt.xlabel("Epochs")
        plt.ylabel("Avg CE Loss")
        plt.ylim(0,1)
        plt.legend(loc="upper right")
        plt.title(f"K Fold: {k}")
    plt.tight_layout()
    # plt.savefig(f"{model}_LoRA_CrossValidation_FT_lr5e-5_50epoch.png", dpi=1200)
    # plt.savefig(f"{model}_DoRA_CrossValidation_FT_lr5e-5_100epoch.png", dpi=1200)
    # plt.show()
        