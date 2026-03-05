import os
import time
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torchvision import transforms

from progress.bar import Bar
from clip import clip

from utils.meter import AverageMeter
from utils.torchtools import load_checkpoint, save_checkpoint, resume_from_checkpoint

from datasets.data import DATASET_GETTERS
from itertools import cycle

from utils.logger_SSL import Logger
import itertools
from copy import deepcopy

from utils.utils import *
from utils.utils import _ECELoss

from collections import Counter
import open_clip
from model import Model, Model_linear, Adapter, AdaptFormer, LoRA
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

best_acc = 0
best_acc1 = 0


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def compute_adjustment_by_py(py, tro, device):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(device)
    return adjustments


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# ============================================================
# CReST 相关：EvalUlbWrapper + PseudoLabelDataset
# ============================================================
class EvalUlbWrapper(Dataset):
    """
    顺序遍历 unlabeled 全体样本，用于收集每个样本 logits。
    要求 __getitem__ 能拿到全局 uidx（最关键）。
    兼容你常见的返回：
      ((u_w,u_s,u_s1), u_real, uidx) 或 (img, _, uidx)
    """
    def __init__(self, ulb_dataset):
        self.ulb = ulb_dataset

    def __len__(self):
        return len(self.ulb)

    def __getitem__(self, i):
        item = self.ulb[i]
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            x0, _, uidx = item[0], item[1], item[2]
            # x0 可能是 (u_w,u_s,u_s1) 或 img
            if isinstance(x0, (tuple, list)):
                img = x0[0]  # u_w
            else:
                img = x0
            return img, 0, int(uidx)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            # fallback：没有 uidx，就用 i（不推荐）
            return item[0], 0, int(i)
        else:
            return item, 0, int(i)

class PseudoLabelFromUlbDataset(Dataset):
    """
    把 unlabeled 中挑出来的样本 + 伪标签 伪装成 labeled 数据项
    输出 (img, y, idx) 以兼容你 train 里 (inputs_x, targets_x, _) 的结构
    """
    def __init__(self, ulb_dataset, indices: np.ndarray, targets: np.ndarray):
        self.ulb = ulb_dataset
        self.indices = np.asarray(indices).astype(np.int64)
        self.targets = np.asarray(targets).astype(np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        y = int(self.targets[k])

        item = self.ulb[i]
        # 解析 img（用 u_w）
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            x0 = item[0]
            if isinstance(x0, (tuple, list)):
                img = x0[0]
            else:
                img = x0
        else:
            img = item
        return img, y, i

# ============================================================
# CReST: dist-align（可选）
# ============================================================
@torch.no_grad()
def dist_align_probs(probs, p_model, p_target, t=0.5, eps=1e-12):
    # probs: [N,C]
    ratio = (p_target + eps) / (p_model + eps)  # [C]
    probs = probs * (ratio.unsqueeze(0) ** t)
    probs = probs / (probs.sum(dim=1, keepdim=True) + eps)
    return probs

class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.num_epochs = cfg.num_epochs
        self.output_dir = cfg.output_dir

        self.cfg = cfg
        self.build_data_loader()
        if cfg.clip_type != '':
            print("Build linear black model")
            self.build_model_linear_black()
        else:
            print("Build peft model")
            self.build_model()
        # self.evaluator = Evaluator(cfg, self.cls_num_list)
        self.best_result = -np.inf
        self._writer = None
        self.th = cfg.th

        # CReST hyperparams
        self.crest_num_gens = getattr(cfg, "crest_num_gens", 5)
        self.crest_alpha = float(getattr(cfg, "crest_alpha", 3.0))
        self.crest_dist_align_t = float(getattr(cfg, "crest_dist_align_t", 0.5))
        self.crest_pro_dist_align = bool(getattr(cfg, "crest_pro_dist_align", True))
        self.crest_epochs_per_gen = int(getattr(cfg, "crest_epochs_per_gen", 6))


        class_list = []
        for i in range(cfg.DATA.NUMBER_CLASSES):
            class_list.append(str(i))

        title = 'PEL-SSL-' + cfg.DATA.NAME
        self.logger = Logger(os.path.join(cfg.output_dir, 'logSSL.txt'), title=title)
        self.logger.set_names(['Top1 acc', 'Best Top1 acc', 'epoch'])

    def build_data_loader(self):
        cfg = self.cfg
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.DATA.NAME](cfg)

        self.lb_base_dataset = labeled_dataset
        self.ulb_base_dataset = unlabeled_dataset
        self.test_dataset = test_dataset

        # 下面这段可以保留（gen0 用）
        self.num_classes = cfg.DATA.NUMBER_CLASSES
        self.classnames = labeled_dataset.classes

        self.train_label_loader = DataLoader(labeled_dataset,
                                             batch_size=cfg.DATA.BATCH_SIZE, num_workers=cfg.DATA.NUM_WORKERS,
                                             shuffle=True, drop_last=True, pin_memory=False, persistent_workers=True)

        self.train_unlabel_loader = DataLoader(unlabeled_dataset,
                                               batch_size=cfg.DATA.BATCH_SIZE * self.cfg.DATA.MU_U,
                                               num_workers=cfg.DATA.NUM_WORKERS, shuffle=True,
                                               drop_last=True, pin_memory=False, persistent_workers=True)

        self.test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=100, pin_memory=False)

    def build_model(self):
        cfg = self.cfg
        # classnames = self.classnames

        print(f"Loading CLIP (backbone: {cfg.backbone})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        print(cfg.prec)

        assert cfg.prec in ["fp16", "fp32", "amp"]
        if cfg.prec == "fp32" or cfg.prec == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        if cfg.template is not None:
            temp = cfg.template
        else:
            temp = "a photo of a {}."
        print(temp)
        print(self.classnames)
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        # prompts = [c for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        print("Building model")
        self.model = Model(cfg, clip_model, self.text_features)
        self.tuner = self.model.tuner
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total params: {total_params}')
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        print(f'Tuned params: {tuned_params}')
        head_params = sum(p.numel() for p in self.tuner.head.parameters())
        tuned_params_without_head = tuned_params - head_params
        print(f'Tuned params (w/o head): {tuned_params_without_head}')

        self.optim = torch.optim.SGD(self.tuner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                     momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, float(cfg.num_epochs))
        self.scaler = GradScaler() if cfg.prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self):
        global best_acc

        if self.cfg.resume:
            directory = self.cfg.resume
            self.start_epoch = self.resume_model_if_exist(directory)

        writer_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        self.time_start = time.time()
        eval_ulb_loader = self._build_eval_ulb_loader()

        # 可选：无监督 loss 权重
        lambda_u = float(getattr(self.cfg, "lambda_u", 1.0))

        it_global = 0
        for gen in range(self.crest_num_gens):
            self.gen = gen
            print(f"\n================ [CReST] Gen {gen}/{self.crest_num_gens-1} ================")

            # progressive dist-align temperature
            if self.crest_pro_dist_align:
                cur = gen / max(self.crest_num_gens - 1, 1)
                cur_dist_align_t = (1.0 - cur) * 1.0 + cur * self.crest_dist_align_t
            else:
                cur_dist_align_t = self.crest_dist_align_t

            # gen>0：把上一代 pseudo 加到 labeled
            if gen > 0 and self.pseudo_label_list is not None:
                picked_idx, picked_y = self._crest_pick_indices(self.pseudo_label_list)
                self._rebuild_lb_loader_with_pseudo(picked_idx, picked_y)
                eval_ulb_loader = self._build_eval_ulb_loader()
                self._reset_optim_sched_amp()

            # ---------------- train this generation ----------------
            for self.epoch in range(self.crest_epochs_per_gen):
                self.tuner.train()

                batch_time = AverageMeter()
                data_time = AverageMeter()
                loss_meter = AverageMeter()
                lx_meter = AverageMeter()
                lu_meter = AverageMeter()
                mask_meter = AverageMeter()

                self.num_batches = self.cfg.total_steps
                label_loader_iter = cycle(self.train_label_loader)
                unlabel_loader_iter = cycle(self.train_unlabel_loader)

                end = time.time()
                bar = Bar(f'Training Gen{gen} Ep{self.epoch+1}', max=self.cfg.total_steps)

                for self.batch_idx in range(self.cfg.total_steps):
                    data_time.update(time.time() - end)

                    (inputs_x, targets_x, _) = next(label_loader_iter)
                    batch_size = inputs_x.shape[0]
                    ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, uidx) = next(unlabel_loader_iter)

                    targets_x = targets_x.to(self.device).long()
                    inputs_x = inputs_x.to(self.device, non_blocking=True)
                    inputs_u_w = inputs_u_w.to(self.device, non_blocking=True)
                    inputs_u_s = inputs_u_s.to(self.device, non_blocking=True)
                    inputs_u_s1 = inputs_u_s1.to(self.device, non_blocking=True)

                    # forward with interleave (BN-friendly)
                    inputs = interleave(
                        torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1), dim=0),
                        3 * self.cfg.DATA.MU_U + 1
                    )

                    feat = self.model(inputs)
                    feat = de_interleave(feat, 3 * self.cfg.DATA.MU_U + 1)

                    logits = self.tuner.head(feat)
                    logits_x = logits[:batch_size]
                    logits_u_w, logits_u_s, logits_u_s1 = logits[batch_size:].chunk(3)

                    # supervised loss
                    Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                    # pseudo label from weak + (optional) dist align
                    with torch.no_grad():
                        probs_w = torch.softmax(logits_u_w.detach(), dim=1)  # [Bu, C]
                        if self.crest_pro_dist_align:
                            probs_w = self._dist_align(probs_w, t=cur_dist_align_t)

                        conf, targets_u = torch.max(probs_w, dim=1)           # [Bu]
                        mask = conf.ge(self.th).float()                       # [Bu] float for weighting

                    # unsupervised loss on two strong views
                    # (Bu,) losses
                    Lu_s  = F.cross_entropy(logits_u_s,  targets_u, reduction='none')
                    Lu_s1 = F.cross_entropy(logits_u_s1, targets_u, reduction='none')
                    Lu = 0.5 * ((Lu_s * mask).mean() + (Lu_s1 * mask).mean())

                    loss = Lx + lambda_u * Lu

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    # meters
                    loss_meter.update(loss.item(), n=batch_size)
                    lx_meter.update(Lx.item(), n=batch_size)
                    lu_meter.update(float(Lu.detach().cpu()), n=batch_size)
                    mask_meter.update(float(mask.mean().detach().cpu()), n=batch_size)

                    current_lr = self.optim.param_groups[0]["lr"]

                    batch_time.update(time.time() - end)
                    end = time.time()

                    it_global += 1
                    n_iter = (gen * self.crest_epochs_per_gen + self.epoch) * self.num_batches + self.batch_idx
                    self._writer.add_scalar("train/loss", loss_meter.avg, n_iter)
                    self._writer.add_scalar("train/Lx", lx_meter.avg, n_iter)
                    self._writer.add_scalar("train/Lu", lu_meter.avg, n_iter)
                    self._writer.add_scalar("train/mask_rate", mask_meter.avg, n_iter)
                    self._writer.add_scalar("train/lr", current_lr, n_iter)

                    # print
                    meet_freq = (self.batch_idx + 1) % self.cfg.print_freq == 0
                    if meet_freq:
                        info = []
                        info += [f"gen [{gen+1}/{self.crest_num_gens}]"]
                        info += [f"ep [{self.epoch+1}/{self.crest_epochs_per_gen}]"]
                        info += [f"step [{self.batch_idx+1}/{self.num_batches}]"]
                        info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                        info += [f"Lx {lx_meter.val:.4f} ({lx_meter.avg:.4f})"]
                        info += [f"Lu {lu_meter.val:.4f} ({lu_meter.avg:.4f})"]
                        info += [f"mask {mask_meter.val:.3f} ({mask_meter.avg:.3f})"]
                        info += [f"t {cur_dist_align_t:.3f}"]
                        info += [f"lr {current_lr:.4e}"]
                        print(" ".join(info))

                    # sched step once per epoch end (keep your old style)
                    if (self.batch_idx + 1) == self.num_batches:
                        self.sched.step()

                    bar.suffix = f'({self.batch_idx+1}/{self.num_batches}) | ' \
                                 f'Loss: {loss_meter.avg:.4f} | Lx: {lx_meter.avg:.4f} | ' \
                                 f'Lu: {lu_meter.avg:.4f} | Mask: {mask_meter.avg:.3f}'
                    bar.next()

                bar.finish()

                # checkpoint per gen-epoch
                last_epoch = (self.epoch + 1) == self.crest_epochs_per_gen
                meet_checkpoint_freq = (
                    ((self.epoch + 1) % self.cfg.checkpoint_freq == 0)
                    if self.cfg.checkpoint_freq > 0 else False
                )
                if meet_checkpoint_freq or last_epoch:
                    self.save_model(self.epoch, self.output_dir)

                # test each epoch (你也可以改成每个 gen 测一次，省时间)
                acc_now = self.test()
                best_acc = max(best_acc, acc_now)
                self.logger.append([acc_now, best_acc, (gen * self.crest_epochs_per_gen + self.epoch + 1)])

            # ---------------- end of generation: build pseudo labels ----------------
            print(f"[CReST] Gen {gen} finished. Building pseudo labels from eval_ulb...")
            ulb_logits_all = self._collect_ulb_logits_all(eval_ulb_loader)     # [N,C] CPU
            self.pseudo_label_list = self._build_pseudo_label_list(ulb_logits_all, cur_dist_align_t)

        print("Finish training")
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        self._writer.close()
        self.logger.close()
   # eval / pseudo label
    # ============================================================
    def _build_eval_ulb_loader(self):
        eval_ds = EvalUlbWrapper(self.ulb_base_dataset)
        return DataLoader(
            eval_ds,
            batch_size=int(getattr(self.cfg, "eval_batch_size_ulb", 256)),
            sampler=SequentialSampler(eval_ds),
            shuffle=False,
            drop_last=False,
            num_workers=getattr(self.cfg, "num_workers", 4),
            pin_memory=True,
        )

    @torch.no_grad()
    def _collect_ulb_logits_all(self, eval_ulb_loader):
        self.tuner.eval()
        N = len(self.ulb_base_dataset)
        logits_all = torch.zeros((N, self.num_classes), dtype=torch.float32)

        for img, _, uidx in tqdm(eval_ulb_loader, desc="eval_ulb", ascii=True):
            img = img.to(self.device, non_blocking=True)
            uidx = torch.as_tensor(uidx).long()

            feat = self.model(img)
            logits = self.tuner.head(feat)  # [B,C]

            logits_all[uidx] = logits.detach().cpu()

        return logits_all

    @torch.no_grad()
    def _build_pseudo_label_list(self, ulb_logits_all_cpu: torch.Tensor, cur_t: float):
        ulb_logits = ulb_logits_all_cpu.to(self.device)
        probs = torch.softmax(ulb_logits, dim=1)  # [N,C]

        # dist-align（可选）
        if self.crest_pro_dist_align:
            if self.p_target is None:
                self.p_target = torch.from_numpy(self.lb_class_dist).to(self.device)
                self.p_target = torch.clamp(self.p_target, min=1e-6)
                self.p_target = self.p_target / self.p_target.sum()

            if self.p_model is None:
                self.p_model = probs.mean(dim=0).detach()
            else:
                self.p_model = 0.999 * self.p_model + 0.001 * probs.mean(dim=0).detach()

            probs = dist_align_probs(probs, self.p_model, self.p_target, t=cur_t)

        conf, pred = torch.max(probs, dim=1)  # [N]
        pseudo_label_list = []
        for c in range(self.num_classes):
            idx = torch.where(pred == c)[0]
            if idx.numel() == 0:
                pseudo_label_list.append(np.array([], dtype=np.int64))
                continue
            order = torch.argsort(conf[idx], descending=True)
            pseudo_label_list.append(idx[order].detach().cpu().numpy())
        return pseudo_label_list

    @torch.no_grad()
    def _dist_align(self, probs, t: float, eps: float = 1e-12):
        """
        probs: [B, C]  (softmax on weak unlabeled logits)
        maintains:
          self.p_target: [C] labeled prior
          self.p_model : [C] EMA unlabeled marginal
        """
        if self.p_target is None:
            p = torch.from_numpy(self.lb_class_dist).to(probs.device).float()
            p = torch.clamp(p, min=1e-6)
            self.p_target = p / p.sum()

        cur = probs.mean(dim=0).detach()
        if self.p_model is None:
            self.p_model = cur
        else:
            self.p_model = 0.999 * self.p_model + 0.001 * cur

        ratio = (self.p_target + eps) / (self.p_model + eps)
        probs = probs * (ratio.unsqueeze(0) ** t)
        probs = probs / (probs.sum(dim=1, keepdim=True) + eps)
        return probs

    # ============================================================
    # crest pick + rebuild labeled loader
    # ============================================================
    def _crest_pick_indices(self, pseudo_label_list):
        """
        按 CReST 公式，从每个类的 pseudo list 里 pick 一部分。
        注意：我用 “rank(按长尾排序)” 而不是 “class id”，更稳。
        """
        lb_dist = self.lb_class_dist  # numpy [C]
        sorted_classes = np.argsort(lb_dist)[::-1]  # major -> minor

        major = lb_dist[sorted_classes[0]]
        minor = lb_dist[sorted_classes[-1]]
        imb_ratio = (minor / (major + 1e-12))
        imb_ratio = max(imb_ratio, 1e-12)
        mu = imb_ratio ** (1.0 / max(self.num_classes - 1, 1))

        picked_idx = []
        picked_y = []

        for rank, c in enumerate(sorted_classes):
            arr = pseudo_label_list[c]
            if arr is None or len(arr) == 0:
                continue

            frac = (mu ** ((self.num_classes - 1) - rank)) ** (1.0 / self.crest_alpha)
            num = int(len(arr) * frac)
            if num <= 0:
                continue

            take = arr[:num]
            picked_idx.append(take)
            picked_y.append(np.full((num,), int(c), dtype=np.int64))
            print(f"[CReST] class {c} rank={rank} pick {num}/{len(arr)} frac={frac:.4f}")

        if len(picked_idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(picked_idx, 0), np.concatenate(picked_y, 0)

    def _rebuild_lb_loader_with_pseudo(self, picked_ulb_idx, picked_targets):
        if picked_ulb_idx is None or len(picked_ulb_idx) == 0:
            print("[CReST] No pseudo labels added. Keep labeled set unchanged.")
            new_lb = self.lb_base_dataset
        else:
            pseudo_ds = PseudoLabelFromUlbDataset(self.ulb_base_dataset, picked_ulb_idx, picked_targets)
            new_lb = ConcatDataset([self.lb_base_dataset, pseudo_ds])

        self.train_label_loader = DataLoader(
            new_lb,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=getattr(self.cfg, "num_workers", 4),
            pin_memory=True,
        )

        # lb_class_dist 一般用 “原始 labeled” 更贴近论文；你也可改成 new_lb 的分布
        self.lb_class_dist = self._get_lb_class_dist_numpy(self.lb_base_dataset)

        print(f"[CReST] Rebuild train_lb_loader done. lb_size={len(new_lb)} (base={len(self.lb_base_dataset)})")

    def _reset_optim_sched_amp(self):
        cfg = self.cfg
        self.optim = torch.optim.SGD(
            self.tuner.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, float(cfg.num_epochs))
        self.scaler = GradScaler() if getattr(cfg, "prec", "fp32") == "amp" else None

    def _get_lb_class_dist_numpy(self, lb_dataset):
        if hasattr(lb_dataset, "targets"):
            y = np.asarray(lb_dataset.targets).astype(np.int64)
            cnt = np.bincount(y, minlength=self.num_classes).astype(np.float64)
            return cnt / (cnt.sum() + 1e-12)

        # fallback：从 loader 统计（通用但慢一点）
        cnt = np.zeros((self.num_classes,), dtype=np.float64)
        tmp_loader = DataLoader(lb_dataset, batch_size=256, shuffle=False, drop_last=False)
        for _, y, *_ in tmp_loader:
            y = np.asarray(y).astype(np.int64).reshape(-1)
            cnt += np.bincount(y, minlength=self.num_classes)
        return cnt / (cnt.sum() + 1e-12)


    @torch.no_grad()
    def test(self):
        self.tuner.eval()
        print(f"Evaluate on the test set")
        preds = np.array([])
        targets = np.array([])
        for batch in tqdm(self.test_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            feat = self.model(image)
            output = self.tuner.head(feat)
            del feat

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())

        targets = targets.astype(int)
        preds = preds.astype(int)
        acc = sum(targets == preds) / len(targets)

        return acc

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        tuner_dict = self.tuner.state_dict()
        optim_dict = self.optim.state_dict()
        sched_dict = self.sched.state_dict()
        save_checkpoint(
            {
                "state_dict": tuner_dict,
                "epoch": epoch + 1,
                "optimizer": optim_dict,
                "scheduler": sched_dict,
                "val_result": val_result
            },
            os.path.join(directory + f'/epoch_{epoch+1}', "tuner"),
            is_best=is_best,
            model_name=model_name,
        )

    def resume_model_if_exist(self, directory):
        file_missing = False

        path = os.path.join(directory, "tuner")
        if not os.path.exists(path):
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        path = os.path.join(directory, "tuner")
        start_epoch = resume_from_checkpoint(
            path, self.tuner, self.optim, self.sched
        )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        model_path = os.path.join(directory, "tuner", model_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path, self.device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        print("Loading weights to {} " 'from "{}" (epoch = {})'.format("tuner", model_path, epoch))
        # set strict=False
        self.tuner.load_state_dict(state_dict, strict=False)

    def load_model_stat(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        model_path = os.path.join(directory, "tuner", model_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path, self.device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        print("Loading weights to {} " 'from "{}" (epoch = {})'.format("tuner", model_path, epoch))
        # set strict=False
        # self.tuner.load_state_dict(state_dict, strict=False)
        return state_dict
