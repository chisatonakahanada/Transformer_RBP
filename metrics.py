import torch
import numpy as np
from sklearn.metrics import average_precision_score, coverage_error

def loss_and_metrics(y_hat, y, loss_fn):
    y = y.float()
    loss = loss_fn(y_hat, y)

    y_prob = torch.sigmoid(y_hat)
    y_pred = (y_prob > 0.5).float()

    label_dict = label_metrics(y, y_pred)
    sample_dict = sample_metrics(y, y_hat)

    return {
        'loss': loss,
        'y_hat': y_hat,
        'y_prob': y_prob,
        'y_pred': y_pred,
        **label_dict,
        **sample_dict
    }

def label_metrics(y, y_pred):
    """ラベル単位の評価指標を計算"""
    acc = (y_pred == y).float().mean()
    per_label_acc = (y_pred == y).float().mean(dim=0)

    tp = ((y_pred == 1) & (y == 1)).float().sum(dim=0)
    fp = ((y_pred == 1) & (y == 0)).float().sum(dim=0)
    tn = ((y_pred == 0) & (y == 0)).float().sum(dim=0)
    fn = ((y_pred == 0) & (y == 1)).float().sum(dim=0)

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    npv = tn / (tn + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return {
        'acc': acc,
        'per_label_acc': per_label_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'npv': npv,
        'f1': f1
    }
   

def sample_metrics(y_true, y_pred_logits):
    """サンプル単位の評価指標を計算"""
    device = y_pred_logits.device
    
    y_true_np = y_true.cpu().numpy()
    y_pred_np = torch.sigmoid(y_pred_logits).detach().cpu().numpy()
    y_pred_bin = (y_pred_np > 0.5).astype(int)

    # Average Precision
    avg_precision = average_precision_score(y_true_np, y_pred_np, average="samples")

    # Ranking Loss
    rank_loss = 0
    num_samples = y_true_np.shape[0]
    for i in range(num_samples):
        true_indices = np.where(y_true_np[i] == 1)[0]
        false_indices = np.where(y_true_np[i] == 0)[0]
        if len(true_indices) == 0 or len(false_indices) == 0:
            continue
        count = 0
        for f in false_indices:
            for t in true_indices:
                if y_pred_np[i, f] > y_pred_np[i, t]:
                    count += 1
        rank_loss += count / (len(true_indices) * len(false_indices))
    rank_loss /= num_samples

    # Hamming Loss
    hamming = np.sum(y_true_np != y_pred_bin) / (y_true_np.shape[0] * y_true_np.shape[1])

    # Coverage
    cov = coverage_error(y_true_np, y_pred_np) - 1

    # One-error
    one_error = 0
    for i in range(num_samples):
        top_label = np.argmax(y_pred_np[i])
        if y_true_np[i, top_label] == 0:
            one_error += 1
    one_error /= num_samples

    # サンプルごとの精度
    intersection = np.logical_and(y_true_np, y_pred_bin).sum(axis=1)
    union = np.logical_or(y_true_np, y_pred_bin).sum(axis=1)
    sample_acc = np.mean(intersection / (union + 1e-10))

    return {
        "sample_avg_precision": avg_precision,
        "ranking_loss": rank_loss,
        "hamming_loss": hamming,
        "coverage": cov,
        "one_error": one_error,
        "sample_accuracy": sample_acc
    }



