import torch
import warnings
warnings.filterwarnings('ignore')

from model import DualStreamConvViT
from dataset import load_eurosat, plot_eda, plot_class_distribution
from train import train_model
from evaluate import evaluate_model, plot_evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATA_DIR = '/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT'  # adjust if needed

train_loader, val_loader, test_loader, CLASS_NAMES, NUM_CLASSES = load_eurosat(DATA_DIR)

# EDA
# full_dataset = ... (you can load once more if needed for distribution plot)
# plot_class_distribution(full_dataset, CLASS_NAMES)
# plot_eda(train_loader, CLASS_NAMES)

model = DualStreamConvViT(
    img_size=64, patch_size=8, embed_dim=256,
    n_heads=8, n_layers=6, num_classes=NUM_CLASSES
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

history, best_val = train_model(model, train_loader, val_loader, device, epochs=30)

all_preds, all_labels, all_probs_np, test_acc, test_top5, test_auc = evaluate_model(
    model, test_loader, device, CLASS_NAMES
)

plot_evaluation(all_preds, all_labels, all_probs_np, history, CLASS_NAMES, test_acc, test_auc)

print("Done!")
print(f"Best Val Acc: {best_val:.4f}")
print(f"Test Top-1: {test_acc:.4f} | Top-5: {test_top5:.4f} | AUC: {test_auc:.4f}")