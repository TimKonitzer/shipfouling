import torch
import torch.nn.functional as F


def soft_target_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    # logits: [B,C], target_probs: [B,C]
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0

    for images, targets, _meta in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = soft_target_loss(logits, targets)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)
