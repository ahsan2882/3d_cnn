from typing import List, Tuple

import torch
from mlxtend.plotting import plot_confusion_matrix  # type: ignore
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from tqdm import tqdm


def test_model(model: torch.nn.Module,
               dataloader: DataLoader[Tuple[torch.Tensor, int]],
               loss_fn: torch.nn.Module,
               class_names: List[str],
               device: torch.device) -> Tuple[float, float]:
    """
    Test the model on the test DataLoader.

    Args:
        model (nn.Module): The model to test.
        dataloader (DataLoader): DataLoader for test data.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to run the model on.

    Returns:
        Tuple[float, float]: Test loss and accuracy.
    """
    model.eval()

    test_loss, test_acc = 0.0, 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.inference_mode():
        for windows, labels in tqdm(dataloader, desc="Testing", unit="batch"):
            windows, labels = windows.to(device), labels.to(device)

            outputs = model(windows)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

            # Get predictions
            y_pred_class = torch.argmax(outputs, dim=1)
            all_preds.append(y_pred_class.cpu())
            all_labels.append(labels.cpu())

    all_pred = torch.cat(all_preds)
    all_label = torch.cat(all_labels)

    # Calculate accuracy
    test_acc = (all_pred == all_label).float().mean().item() * 100
    test_loss /= len(dataloader)
    confusion_matrix_plot(all_label, all_pred, class_names)
    return test_loss, test_acc


def confusion_matrix_plot(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: List[str]) -> None:
    """
    Plots the confusion matrix.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.
    """
    cm = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    cm_tensor = cm(preds=y_pred, target=y_true)

    plot_confusion_matrix(conf_mat=cm_tensor.numpy(),
                          class_names=class_names, figsize=(10, 7))
