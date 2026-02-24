import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

def evaluate_model(model, test_loader, device, CLASS_NAMES):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_probs_np = np.array(all_probs)
    test_acc = accuracy_score(all_labels, all_preds)
    test_top5 = top_k_accuracy_score(all_labels, all_probs_np, k=5)
    test_auc = roc_auc_score(all_labels, all_probs_np, multi_class='ovr', average='macro')

    print(f"Test Top-1: {test_acc:.4f} | Top-5: {test_top5:.4f} | AUC-ROC: {test_auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    return all_preds, all_labels, all_probs_np, test_acc, test_top5, test_auc


def plot_evaluation(all_preds, all_labels, all_probs_np, history, CLASS_NAMES, test_acc, test_auc):
    mean_n = np.array([0.3444, 0.3803, 0.4078])
    std_n  = np.array([0.2028, 0.1368, 0.1153])

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35)
    fig.suptitle('Evaluation — DualStream-ConvViT (EuroSAT)')

    # Confusion Matrix (normalized)
    ax_cm = fig.add_subplot(gs[0, :2])
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax_cm)
    ax_cm.set_title('Normalized Confusion Matrix')

    # Per-class accuracy
    ax_cls = fig.add_subplot(gs[0, 2:])
    per_class_acc = cm_norm.diagonal()
    ax_cls.barh(CLASS_NAMES, per_class_acc)
    ax_cls.set_title('Per-Class Accuracy')
    ax_cls.axvline(test_acc, color='red', linestyle='--', label=f'Overall {test_acc:.3f}')
    ax_cls.legend()

    # Training curves
    ep = range(1, len(history['train_loss']) + 1)
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.plot(ep, history['train_loss'], label='Train Loss')
    ax_loss.plot(ep, history['val_loss'], label='Val Loss')
    ax_loss.set_title('Loss Curves')
    ax_loss.legend()

    ax_acc = fig.add_subplot(gs[1, 1])
    ax_acc.plot(ep, [a for a in history['val_acc']], label='Val Acc')
    ax_acc.set_title('Validation Accuracy')
    ax_acc.legend()

    # ROC
    ax_roc = fig.add_subplot(gs[1, 2:])
    y_bin = label_binarize(all_labels, classes=range(len(CLASS_NAMES)))
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs_np[:, i])
        ax_roc.plot(fpr, tpr, label=f'{CLASS_NAMES[i][:10]}')
    ax_roc.set_title(f'ROC (macro AUC = {test_auc:.3f})')
    ax_roc.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('assets/eurosat_evaluation_fixed.png', dpi=200, bbox_inches='tight')
    plt.show()