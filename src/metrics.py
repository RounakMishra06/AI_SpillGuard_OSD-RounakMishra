# metrics.py
import torch

def threshold(logits, thresh=0.5):
    """Apply sigmoid and threshold to logits"""
    return (torch.sigmoid(logits) > thresh).float()

def iou(preds, targets, eps=1e-6):
    """Intersection over Union (IoU) metric"""
    intersection = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - intersection
    iou_score = (intersection + eps) / (union + eps)
    return iou_score.mean().item()

def dice(preds, targets, eps=1e-6):
    """Dice coefficient metric"""
    intersection = (preds * targets).sum(dim=(2,3))
    total = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice_score = (2 * intersection + eps) / (total + eps)
    return dice_score.mean().item()

def pixel_accuracy(preds, targets):
    """Pixel-wise accuracy"""
    correct = (preds == targets).float()
    return correct.mean().item()

def precision_recall(preds, targets, eps=1e-6):
    """Compute precision and recall"""
    tp = (preds * targets).sum(dim=(2,3))
    fp = (preds * (1 - targets)).sum(dim=(2,3))
    fn = ((1 - preds) * targets).sum(dim=(2,3))
    
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    
    return precision.mean().item(), recall.mean().item()