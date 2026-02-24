import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

IMG_SIZE = 64
BATCH_SIZE = 128
SEED = 42

train_transform = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.2028, 0.1368, 0.1153])
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.2028, 0.1368, 0.1153])
])

def load_eurosat(data_dir):
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    CLASS_NAMES = full_dataset.classes
    NUM_CLASSES = len(CLASS_NAMES)

    n = len(full_dataset)
    n_test = int(0.15 * n)
    n_val = int(0.15 * n)
    n_train = n - n_test - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, CLASS_NAMES, NUM_CLASSES


def plot_eda(train_loader, CLASS_NAMES):
    mean_n = np.array([0.3444, 0.3803, 0.4078])
    std_n  = np.array([0.2028, 0.1368, 0.1153])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('EuroSAT — Sample Images per Class')

    class_samples = {}
    for imgs, labels in train_loader:
        for img, lbl in zip(imgs, labels):
            l = lbl.item()
            if l not in class_samples:
                class_samples[l] = img
        if len(class_samples) == len(CLASS_NAMES):
            break

    for i in range(min(10, len(CLASS_NAMES))):
        ax = axes[i // 5, i % 5]
        if i in class_samples:
            img_show = (class_samples[i].numpy().transpose(1,2,0) * std_n + mean_n).clip(0,1)
            ax.imshow(img_show)
        ax.set_title(CLASS_NAMES[i], fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('assets/eurosat_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_class_distribution(full_dataset, CLASS_NAMES):
    if hasattr(full_dataset, 'targets'):
        class_counts = np.bincount(full_dataset.targets)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(CLASS_NAMES, class_counts)
        ax.set_title('Class Distribution — EuroSAT')
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()