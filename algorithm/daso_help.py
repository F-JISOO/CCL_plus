# daso_helper.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_ce_loss(logits, soft_targets, reduction="mean"):
    """
    logits: [B, C]
    soft_targets: [B, C] (sum=1)
    """
    logp = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * logp).sum(dim=1)  # [B]
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class DASOFeatureQueue(nn.Module):
    """
    class-wise ring buffer queue, store normalized features.
    prototypes = mean feature per class (normalized).
    """
    def __init__(self, num_classes: int, feat_dim: int, queue_length: int = 256, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.queue_length = queue_length
        self.device = device

        bank = torch.zeros(num_classes, queue_length, feat_dim, dtype=torch.float32)
        ptr = torch.zeros(num_classes, dtype=torch.long)
        cnt = torch.zeros(num_classes, dtype=torch.long)

        self.register_buffer("bank", bank)
        self.register_buffer("ptr", ptr)
        self.register_buffer("cnt", cnt)

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        feats: [B, D]
        labels: [B]
        """
        if feats.numel() == 0:
            return
        feats = F.normalize(feats.float(), dim=1)
        labels = labels.long()

        for c in labels.unique():
            c = int(c.item())
            idx = torch.where(labels == c)[0]
            if idx.numel() == 0:
                continue
            f = feats[idx]  # [n, D]
            n = f.size(0)

            p = int(self.ptr[c].item())
            # write into ring
            if n >= self.queue_length:
                f = f[-self.queue_length:]
                n = f.size(0)

            end = p + n
            if end <= self.queue_length:
                self.bank[c, p:end] = f
            else:
                first = self.queue_length - p
                self.bank[c, p:] = f[:first]
                self.bank[c, :end - self.queue_length] = f[first:]

            self.ptr[c] = (p + n) % self.queue_length
            self.cnt[c] = torch.clamp(self.cnt[c] + n, max=self.queue_length)

    @torch.no_grad()
    def get_prototypes(self):
        """
        returns:
          prototypes: [C, D] normalized
          valid_mask: [C] bool (cnt>0)
        """
        valid = self.cnt > 0
        proto = torch.zeros(self.num_classes, self.feat_dim, device=self.bank.device, dtype=torch.float32)
        for c in range(self.num_classes):
            k = int(self.cnt[c].item())
            if k <= 0:
                continue
            proto[c] = self.bank[c, :k].mean(dim=0)
        proto = F.normalize(proto, dim=1)
        return proto, valid


@torch.no_grad()
def daso_make_pseudo_probs(
    probs_cls,                # [B,C]  softmax(logits_u_w)
    probs_sim,                # [B,C]  softmax(sim_w/T_proto)
    interp_alpha: float = 0.3,
    with_dist_aware: bool = True,
    pseudo_label_dist=None,   # [C] EMA
    T_dist: float = 1.5,
    eps: float = 1e-12,
):
    """
    DASO: mix classifier prob and semantic (prototype) prob
      p_mix = (1-a)*p_cls + a*p_sim
    dist-aware (paper): reweight by inverse dist^(T_dist)
      w_c ∝ (p_pl_dist[c])^{-T_dist}
      p_mix <- normalize(p_mix * w)
    """
    p_mix = (1.0 - interp_alpha) * probs_cls + interp_alpha * probs_sim  # [B,C]

    if with_dist_aware and (pseudo_label_dist is not None):
        dist = torch.clamp(pseudo_label_dist, min=eps)
        w = dist.pow(-T_dist)  # inverse power
        w = w / (w.sum() + eps)
        p_mix = p_mix * w.unsqueeze(0)
        p_mix = p_mix / (p_mix.sum(dim=1, keepdim=True) + eps)

    return p_mix


@torch.no_grad()
def ema_update_dist(old_dist, new_dist, momentum=0.999):
    """
    old_dist/new_dist: [C] sum=1
    """
    if old_dist is None:
        return new_dist
    return momentum * old_dist + (1.0 - momentum) * new_dist
