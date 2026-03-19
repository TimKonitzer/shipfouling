import torch
import torch.nn.functional as F


def soft_target_loss(logits: torch.Tensor, target_probs: torch.Tensor, class_weights: torch.Tensor = None) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    
    kl = F.kl_div(log_probs, target_probs, reduction="none")
    batch_kl = kl.sum(dim=1)
    
    if class_weights is not None:
        sample_weights = (target_probs * class_weights).sum(dim=1)
        batch_kl = batch_kl * sample_weights
        
    return batch_kl.mean()


def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0.0
    n = 0

    if class_weights is not None:
        class_weights = class_weights.to(device)

    for images, targets, _meta in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = soft_target_loss(logits, targets, class_weights)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)
