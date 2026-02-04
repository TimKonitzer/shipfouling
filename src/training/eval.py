import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, targets, _meta in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, targets, reduction="batchmean")

        pred = logits.argmax(dim=1)
        true = targets.argmax(dim=1)  # majority proxy from soft labels

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((pred == true).sum().item())
        total += images.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }
