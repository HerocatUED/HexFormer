import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension with respect to sorted errors
    See Alg. 1 in paper
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels):
    """
    Multi-class Lovasz-Softmax loss
    probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1).
    labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if fg.sum() == 0:
            # only background, class does not exist
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return sum(losses) / len(losses)


class LovaszSoftmax(nn.Module):
    def __init__(self):
        super(LovaszSoftmax, self).__init__()

    def forward(self, probas, labels):
        probas = probas.view(-1, probas.size(-1))  # Flatten predictions
        labels = labels.view(-1)  # Flatten labels
        return lovasz_softmax_flat(probas, labels)


if __name__ == "__main__":
    N = 1000  # Number of points
    C = 5  # Number of classes
    logits = torch.randn(N, C, requires_grad=True)  # Example logits from a network
    labels = torch.randint(0, C, (N,))  # Example ground truth labels
    criterion = LovaszSoftmax()
    loss = criterion(F.softmax(logits, dim=1), labels)
    loss.backward()
    print(f"Loss: {loss.item()}")
