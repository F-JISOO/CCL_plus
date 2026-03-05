import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
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
            self.build_linear_black_model()
        else:
            print("Build peft model")
            self.build_model()
        # self.evaluator = Evaluator(cfg, self.cls_num_list)
        self.best_result = -np.inf
        self._writer = None
        self.th = cfg.th

        class_list = []
        for i in range(cfg.DATA.NUMBER_CLASSES):
            class_list.append(str(i))

        title = 'PEL-SSL-' + cfg.DATA.NAME
        self.logger = Logger(os.path.join(cfg.output_dir, 'logSSL.txt'), title=title)
        self.logger.set_names(['Top1 acc', 'Best Top1 acc', 'epoch'])

        # ---- curve buffers (one value per epoch) ----
        self.curve_epoch = []
        self.curve_test_acc = []
        self.curve_entropy = []
        self.curve_n_accept = []
        self.curve_total_accept = []

        # =========================
        # CPG init (add in __init__)
        # =========================
        self.warm_up     = int(getattr(cfg, "warm_up", 5))       # CPG warmup epoch
        self.memory_step = int(getattr(cfg, "memory_step", 5))   # CPG memory/update step
        self.p_cutoff    = float(getattr(cfg, "th", 0.95))       # use your th
        self.smoothing   = float(getattr(cfg, "smoothing", 0.0)) # label smoothing
        self.uratio      = int(getattr(cfg.DATA, "MU_U", 1))     # unlabeled ratio (batch_size_u = uratio*bs)

        # labeled class-count dist: [C]
        cnt = torch.zeros(self.num_classes, device=self.device, dtype=torch.float)
        for batch in self.train_label_loader:
            y = batch[1].to(self.device).long()
            cnt += torch.bincount(y, minlength=self.num_classes).float()
        self.lb_dist = cnt.detach()  # counts

        # selected-unlabeled pseudo-label dist: [C]
        self.select_ulb_dist = torch.zeros(self.num_classes, device=self.device, dtype=torch.float)
        self.lb_select_ulb_dist = (self.lb_dist + self.select_ulb_dist).detach()

        # memory buffers (store on CPU to save GPU)
        self.select_ulb_idx = None          # LongTensor [N] on CPU
        self.select_ulb_pseudo = None       # LongTensor [N] on CPU
        self.select_ulb_true = None         # LongTensor [N] on CPU (optional, if u_real exists)

        # feature augmentation states (CPG)
        self.optim_cfc = None     # class feature center [C, D]
        self.feat_aug_r = None    # per-class scalar [C]
        self.fd = None            # feature dim D

    def build_data_loader(self):
        cfg = self.cfg
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.DATA.NAME](
            cfg)

        self.num_classes = cfg.DATA.NUMBER_CLASSES
        self.classnames = labeled_dataset.classes

        # self.sampled_cls_num_list = self.cls_num_list
        # ------ test ------
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
    
    def build_linear_black_model(self):
        cfg = self.cfg
        # classnames = self.classnames

        print(f"Loading CLIP (backbone: {cfg.backbone})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=cfg.clip_type)
        clip_model.half()
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
        self.model = Model_linear(cfg, clip_model, self.text_features)
        self.tuner = self.model.tuner
        self.clip_model = clip_model
        # self.dtype = clip_model.dtype
        
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
    
    def _cpg_epoch_hook(self):
        # warmup: do NOT use selected dist; just stabilize training
        if self.epoch < self.warm_up:
            self.lb_select_ulb_dist = self.lb_dist.detach()
            self.select_ulb_dist = torch.ones_like(self.select_ulb_dist)  # avoid zeros
            return

        # at epoch == warm_up: build feature center + feat_aug_r once
        if self.epoch == self.warm_up:
            self.lb_select_ulb_dist = self.lb_dist.detach()
            self.select_ulb_dist = torch.ones_like(self.select_ulb_dist)
            self._cpg_recompute_centers_and_aug_r()
            return

        # every memory_step epochs: update select_ulb_dist from memory, reset memory, recompute centers
        if (self.memory_step > 0) and (self.epoch % self.memory_step == 0):
            self._cpg_memory_update()
            self._cpg_recompute_centers_and_aug_r()
    
    def _soft_ce(self, logits, soft_targets):
        # logits: [B,C], soft_targets: [B,C]
        logp = F.log_softmax(logits, dim=-1)
        return -(soft_targets * logp).sum(dim=-1).mean()

    def _make_smooth_onehot(self, y, C, smoothing, device, dtype):
        # y: [B] long
        if smoothing <= 0:
            return None
        oh = torch.zeros((y.size(0), C), device=device, dtype=dtype)
        oh.fill_(smoothing / (C - 1))
        oh.scatter_(1, y.view(-1, 1), 1.0 - smoothing)
        return oh

    def _cpg_logit_adjust(self, logits, dist_counts):
        # dist_counts: [C] float counts
        adj = torch.log(dist_counts / dist_counts.sum().clamp(min=1e-12)).to(
            device=logits.device, dtype=logits.dtype
        ).view(1, -1)
        return logits + adj

    @torch.no_grad()
    def _cpg_refine_select_dist(self):
        # CPG: refined_select_ulb_dist = replace too-small counts with the 2nd-smallest non-zero
        lb_min = float(self.lb_dist.min().item()) if self.lb_dist.numel() > 0 else 0.0
        before = torch.where(self.select_ulb_dist <= lb_min, torch.zeros_like(self.select_ulb_dist), self.select_ulb_dist)

        uniq = torch.unique(before)
        uniq_sorted, _ = torch.sort(uniq)

        if uniq_sorted.numel() <= 1:
            refined = torch.ones_like(before)
        else:
            refined = torch.where(before == 0, uniq_sorted[1], before)
        return refined

    @torch.no_grad()
    def _cpg_memory_update(self):
        """
        CPG memory cleaning (same rule as repo):
        for each uidx, if it has >12 records and majority label >0.8, accept it;
        then update select_ulb_dist from accepted pseudo-labels.
        """
        if (self.select_ulb_idx is None) or (self.select_ulb_pseudo is None):
            # nothing selected
            self.select_ulb_dist.zero_()
            self.lb_select_ulb_dist = (self.lb_dist + self.select_ulb_dist).detach()
            return

        idx = self.select_ulb_idx
        pseudo = self.select_ulb_pseudo

        # group by idx
        from collections import defaultdict, Counter
        mp = defaultdict(list)
        for i, p in zip(idx.tolist(), pseudo.tolist()):
            mp[i].append(p)

        accepted = []
        for i, plist in mp.items():
            if len(plist) <= 12:
                continue
            most_common, n = Counter(plist).most_common(1)[0]
            if n > 0.8 * len(plist):
                accepted.append(most_common)

        if len(accepted) == 0:
            self.select_ulb_dist.zero_()
        else:
            acc = torch.tensor(accepted, device=self.device, dtype=torch.long)
            self.select_ulb_dist = torch.bincount(acc, minlength=self.num_classes).float()

        self.lb_select_ulb_dist = (self.lb_dist + self.select_ulb_dist).detach()

        # reset memory
        self.select_ulb_idx = None
        self.select_ulb_pseudo = None
        self.select_ulb_true = None

    @torch.no_grad()
    def _cpg_recompute_centers_and_aug_r(self):
        """
        CPG: compute class feature centers + feat_aug_r using current labeled loader.
        """
        C = self.num_classes
        # feature dim
        # (use a small forward to infer D)
        for batch in self.train_label_loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device).long()
            feat = self.model(x)
            D = int(feat.size(1))
            self.fd = D
            break

        # class center
        center = torch.zeros((C, self.fd), device=self.device, dtype=torch.float32)
        for batch in self.train_label_loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device).long()
            feat = self.model(x).detach().float()
            center.index_add_(0, y, feat)
        denom = self.lb_select_ulb_dist.to(center.device).clamp(min=1.0).view(-1, 1)
        center = center / denom
        self.optim_cfc = center  # [C,D] float32

        # sim_num accumulation
        sim_num = torch.zeros((C,), device=self.device, dtype=torch.float32)
        for batch in self.train_label_loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device).long()
            feat = self.model(x).detach().float()
            # cosine sim in [0,1]
            sim = (F.cosine_similarity(F.normalize(feat, dim=1), F.normalize(center[y], dim=1), dim=1) + 1.0) / 2.0
            sim_num.index_add_(0, y, sim)

        sim_mean = sim_num / self.lb_select_ulb_dist.to(sim_num.device).clamp(min=1.0)
        sim_num_inv = 1.0 / sim_mean.clamp(min=1e-6)
        feat_aug_r = sim_num_inv / sim_num_inv.sum().clamp(min=1e-12)

        self.feat_aug_r = feat_aug_r.to(device=self.device, dtype=torch.float16)  # keep fp16 for speed

    def _cpg_step_loss(self, x_lb_w, y_lb, x_ulb_w, x_ulb_s, uidx=None, u_real=None):
        """
        Fully replace FixMatch loss with CPG loss:
        total_loss = sup_loss(main head with LA on updated dist) + aux_loss(aux head)
        and do CPG selection (agreement + confidence) to update memory buffers.
        """
        bs = x_lb_w.size(0)
        C = self.num_classes

        # forward -> features
        # NOTE: follow your SCAD-style interleave if you need BN sync; here keep simple
        feat_lb_w = self.model(x_lb_w)
        feat_ulb_w = self.model(x_ulb_w)
        feat_ulb_s = self.model(x_ulb_s)

        if self.fd is None:
            self.fd = int(feat_lb_w.size(1))

        # logits: main / aux
        logits_lb_w  = self.tuner.head(feat_lb_w)
        logits_ulb_w = self.tuner.head(feat_ulb_w)
        logits_ulb_s = self.tuner.head(feat_ulb_s)

        aux_lb_w  = self.tuner.head1(feat_lb_w)
        aux_ulb_w = self.tuner.head1(feat_ulb_w)
        aux_ulb_s = self.tuner.head1(feat_ulb_s)

        # --------- supervised main loss (warmup: CE on labeled; after warmup: feature augmentation) ---------
        if self.epoch < self.warm_up:
            # label smoothing (soft CE)
            y_soft = self._make_smooth_onehot(y_lb.long(), C, self.smoothing, logits_lb_w.device, logits_lb_w.dtype)
            logits_adj = self._cpg_logit_adjust(logits_lb_w, self.lb_select_ulb_dist)
            sup_loss = self._soft_ce(logits_adj, y_soft) if y_soft is not None else F.cross_entropy(logits_adj, y_lb.long())
        else:
            # feature augmentation (CPG) using feat_aug_r + oversampling factor
            # early after warmup: still use smoothing; later use adaptive times
            if (self.feat_aug_r is None) or (self.optim_cfc is None):
                # safety fallback
                logits_adj = self._cpg_logit_adjust(logits_lb_w, self.lb_select_ulb_dist)
                sup_loss = F.cross_entropy(logits_adj, y_lb.long())
            else:
                if self.epoch < self.warm_up + self.memory_step:
                    aug_times = torch.full_like(y_lb.long(), 10, device=y_lb.device)
                    y_soft = self._make_smooth_onehot(y_lb.long(), C, self.smoothing, logits_lb_w.device, logits_lb_w.dtype)
                    base_labels = y_soft if y_soft is not None else y_lb.long()
                else:
                    # times = int( 10 * quantile(lb_select_ulb_dist) / lb_select_ulb_dist[class] )
                    q = torch.sort(self.lb_select_ulb_dist).values[C // 3].clamp(min=1.0)
                    times_per_class = (10.0 * q / self.lb_select_ulb_dist.clamp(min=1.0)).int().clamp(min=0)
                    aug_times = times_per_class[y_lb.long()].to(device=y_lb.device)
                    base_labels = y_lb.long()

                aug_feats = feat_lb_w
                if isinstance(base_labels, torch.Tensor) and base_labels.dim() == 2:
                    aug_labels = base_labels
                    base_cls = base_labels.argmax(dim=1)
                else:
                    aug_labels = base_labels
                    base_cls = y_lb.long()

                # loop-augment (match CPG behavior)
                for i, (feat_i, cls_i, t_i) in enumerate(zip(feat_lb_w, base_cls, aug_times)):
                    t = int(t_i.item())
                    if t <= 0:
                        continue
                    # noise scale r(class)
                    r = self.feat_aug_r[cls_i].to(device=feat_lb_w.device, dtype=feat_lb_w.dtype).view(1, 1)
                    feat_norm = F.normalize(feat_i.unsqueeze(0), dim=1)
                    noise = torch.randn((t, self.fd), device=feat_lb_w.device, dtype=feat_lb_w.dtype) * r * feat_norm
                    aug_feats = torch.cat([aug_feats, feat_i.unsqueeze(0).repeat(t, 1) + noise], dim=0)

                    if aug_labels.dim() == 2:
                        aug_labels = torch.cat([aug_labels, base_labels[i].unsqueeze(0).repeat(t, 1)], dim=0)
                    else:
                        aug_labels = torch.cat([aug_labels, cls_i.view(1).repeat(t)], dim=0)

                aug_logits = self.tuner.head(aug_feats)
                aug_logits_adj = self._cpg_logit_adjust(aug_logits, self.lb_select_ulb_dist)
                if aug_labels.dim() == 2:
                    sup_loss = self._soft_ce(aug_logits_adj, aug_labels)
                else:
                    sup_loss = F.cross_entropy(aug_logits_adj, aug_labels.long())

        # --------- aux loss (CPG) ---------
        if self.epoch < self.warm_up:
            # aux pseudo from aux_ulb_w (hard), then smooth target on aux_ulb_s
            aux_pseudo = aux_ulb_w.detach().softmax(dim=-1).argmax(dim=-1)
            aux_soft = self._make_smooth_onehot(aux_pseudo, C, self.smoothing, aux_ulb_s.device, aux_ulb_s.dtype)
            loss_ulb_aux = self._soft_ce(aux_ulb_s, aux_soft) if aux_soft is not None else F.cross_entropy(aux_ulb_s, aux_pseudo)

            # labeled aux also uses LA + smoothing
            y_soft = self._make_smooth_onehot(y_lb.long(), C, self.smoothing, aux_lb_w.device, aux_lb_w.dtype)
            aux_lb_adj = self._cpg_logit_adjust(aux_lb_w, self.lb_select_ulb_dist)
            loss_lb_aux = self._soft_ce(aux_lb_adj, y_soft) if y_soft is not None else F.cross_entropy(aux_lb_adj, y_lb.long())
            aux_loss = loss_ulb_aux + loss_lb_aux

            util_ratio = 0.0
            total_loss = sup_loss + aux_loss
            return total_loss, util_ratio, None, None

        # after warmup: selection uses main head with refined_select_ulb_dist
        refined = self._cpg_refine_select_dist()
        probs_w = (logits_ulb_w + torch.log(refined / refined.sum().clamp(min=1e-12)).to(logits_ulb_w.device).view(1, -1)).detach().softmax(dim=-1)
        probs_s = (logits_ulb_s + torch.log(refined / refined.sum().clamp(min=1e-12)).to(logits_ulb_s.device).view(1, -1)).detach().softmax(dim=-1)

        pseudo_w = probs_w.argmax(dim=-1)
        pseudo_s = probs_s.argmax(dim=-1)

        conf_w = probs_w.max(dim=-1).values
        conf_s = probs_s.max(dim=-1).values

        th_eff = self.p_cutoff * (1.0 - self.smoothing)
        mask = (conf_w >= th_eff) & (conf_s >= th_eff) & (pseudo_w == pseudo_s)

        # aux: unlabeled aux CE + labeled aux CE (labeled uses LA)
        aux_pseudo = aux_ulb_w.detach().softmax(dim=-1).argmax(dim=-1)
        loss_ulb_aux = F.cross_entropy(aux_ulb_s, aux_pseudo)

        aux_lb_adj = self._cpg_logit_adjust(aux_lb_w, self.lb_select_ulb_dist)
        loss_lb_aux = F.cross_entropy(aux_lb_adj, y_lb.long())
        aux_loss = loss_ulb_aux + loss_lb_aux

        # update memory buffers
        if uidx is not None:
            uidx_cpu = uidx.detach().cpu().long()
            pseudo_cpu = pseudo_w.detach().cpu().long()
            if u_real is not None and torch.is_tensor(u_real):
                true_cpu = u_real.detach().cpu().long()
            else:
                true_cpu = None

            sel = mask.detach().cpu()
            if sel.any():
                if self.select_ulb_idx is None:
                    self.select_ulb_idx = uidx_cpu[sel]
                    self.select_ulb_pseudo = pseudo_cpu[sel]
                    self.select_ulb_true = true_cpu[sel] if true_cpu is not None else None
                else:
                    self.select_ulb_idx = torch.cat([self.select_ulb_idx, uidx_cpu[sel]], dim=0)
                    self.select_ulb_pseudo = torch.cat([self.select_ulb_pseudo, pseudo_cpu[sel]], dim=0)
                    if true_cpu is not None:
                        if self.select_ulb_true is None:
                            self.select_ulb_true = true_cpu[sel]
                        else:
                            self.select_ulb_true = torch.cat([self.select_ulb_true, true_cpu[sel]], dim=0)

        util_ratio = float(mask.float().mean().item())
        total_loss = sup_loss + aux_loss
        return total_loss, util_ratio, pseudo_w.detach(), mask.detach()

    def train(self):
        global best_acc

        if self.cfg.resume:
            directory = self.cfg.resume
            self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        self.w_con = self.cfg.w_con

        self.time_start = time.time()

        for self.epoch in range(self.start_epoch, self.num_epochs):
            self.tuner.train()
            self._cpg_epoch_hook()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            self.num_batches = self.cfg.total_steps
            label_loader_iter = cycle(self.train_label_loader)
            unlabel_loader_iter = cycle(self.train_unlabel_loader)

            pseudo_count = torch.zeros(self.num_classes, dtype=torch.long)  # CPU
            n_accept = 0    
            end = time.time()

            bar = Bar('Training', max=self.cfg.total_steps)
            for self.batch_idx in range(self.cfg.total_steps):
                data_time.update(time.time() - end)

                (inputs_x, targets_x, _) = next(label_loader_iter)
                batch_size = inputs_x.shape[0]
                ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, uidx) = next(unlabel_loader_iter)
                uidx = uidx.to(self.device)

                targets_x = targets_x.to(self.device)
                targets_x = targets_x.to(torch.long)


                inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_x.to(self.device), inputs_u_w.to(self.device), \
                                                                inputs_u_s.to(self.device), inputs_u_s1.to(self.device)

                loss, util_ratio, targets_u, mask = self._cpg_step_loss(
                    inputs_x, targets_x,
                    inputs_u_w, inputs_u_s,
                    uidx=uidx, u_real=u_real
                )
                if mask is not None and mask.any():
                    y = targets_u[mask].detach().cpu()
                    pseudo_count += torch.bincount(y, minlength=self.num_classes)
                    n_accept += int(mask.sum().item())
                


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                current_logit_scale = self.tuner.head.logit_scale.item()
                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())

                batch_time.update(time.time() - end)
                meet_freq = (self.batch_idx + 1) % self.cfg.print_freq == 0
                only_few_batches = self.num_batches < self.cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += self.num_batches - self.batch_idx - 1
                    nb_remain += (
                                         self.num_epochs - self.epoch - 1
                                 ) * self.num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.num_epochs}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"s {current_logit_scale:.4f}"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = self.epoch * self.num_batches + self.batch_idx
                self._writer.add_scalar("train/loss", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/lr", current_lr, n_iter)

                if (self.batch_idx + 1) == self.num_batches:
                    self.sched.step()

                end = time.time()
                bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                            'Loss: {loss:.4f} '.format(
                    batch=self.batch_idx + 1,
                    size=self.cfg.total_steps,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss_meter.avg,
                )
                bar.next()
            bar.finish()
            last_epoch = (self.epoch + 1) == self.num_epochs
            meet_checkpoint_freq = (
                (self.epoch + 1) % self.cfg.checkpoint_freq == 0
                if self.cfg.checkpoint_freq > 0 else False
            )

            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

            # torch.cuda.empty_cache()

            acc_now = self.test()

            # ---- compute class-distribution entropy for pseudo labels ----
            eps = 1e-12
            total_accept = int(pseudo_count.sum().item())
            if total_accept > 0:
                q = pseudo_count.float() / (total_accept + eps)
                entropy = float(-(q * (q + eps).log()).sum().item())
            else:
                entropy = float("nan")  # 或者 0.0，但 nan 更能暴露“没接纳到伪标签”

            # ---- append to vectors ----
            self.curve_epoch.append(self.epoch + 1)
            self.curve_test_acc.append(float(acc_now))
            self.curve_entropy.append(float(entropy))
            self.curve_n_accept.append(int(n_accept))
            self.curve_total_accept.append(int(total_accept))


            best_acc = max(best_acc, acc_now)
            self.logger.append([acc_now, best_acc, self.epoch + 1])

        print("Finish training")

        print("Deploy the last-epoch model for testing")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self._writer.close()
        curve_dir = os.path.join(self.output_dir, "curves")
        os.makedirs(curve_dir, exist_ok=True)
        curve_path = os.path.join(curve_dir, "curves.pt")

        pack = {
            "epoch": torch.tensor(self.curve_epoch, dtype=torch.long),
            "test_acc": torch.tensor(self.curve_test_acc, dtype=torch.float),
            "pseudo_entropy": torch.tensor(self.curve_entropy, dtype=torch.float),
            "pseudo_n_accept": torch.tensor(self.curve_n_accept, dtype=torch.long),
            "pseudo_total_accept": torch.tensor(self.curve_total_accept, dtype=torch.long),
            "num_classes": int(self.num_classes),
            "threshold": float(self.th),
            "method": "fixmatch",  # 你也可以写 cfg.method
        }
        torch.save(pack, curve_path)
        print(f"[Saved] curves -> {curve_path}")

        self.logger.close()

    def train_black_model(self):
        global best_acc
        global best_zs_acc
        global best_acc1

        if self.cfg.resume:
            directory = self.cfg.resume
            self.start_epoch = self.resume_model_if_exist(directory)

        self.tuner.head1.weight.data = self.tuner.head.weight.data.clone()
        # Initialize summary writer
        writer_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        self.w_con = self.cfg.w_con
        ulab_len = len(self.train_unlabel_loader.dataset)

        self.alpha = self.cfg.alpha
        self.betabase = torch.ones((self.num_classes, self.num_classes)).to(self.device)
        self.betabase[torch.arange(self.num_classes), torch.arange(self.num_classes)] = 0.0

        self.smoothing = self.cfg.smoothing
        self.th_min = self.cfg.th_min

        self.time_start = time.time()

        for self.epoch in range(self.start_epoch, self.num_epochs):
            self.tuner.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            self.num_batches = self.cfg.total_steps
            label_loader_iter = cycle(self.train_label_loader)
            unlabel_loader_iter = cycle(self.train_unlabel_loader)

            selected_label = torch.ones((ulab_len,), dtype=torch.long, ) * -1
            selected_label = selected_label.to(self.device)
            classwise_acc = torch.zeros((self.num_classes,)).to(self.device)
            end = time.time()

            for self.batch_idx in range(self.cfg.total_steps):
                data_time.update(time.time() - end)

                (inputs_x, targets_x, _) = next(label_loader_iter)
                batch_size = inputs_x.shape[0]
                ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, uidx) = next(unlabel_loader_iter)

                uidx = uidx.to(self.device)
                targets_x = targets_x.to(self.device)
                targets_x = targets_x.to(torch.long)


                inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_x.to(self.device), inputs_u_w.to(self.device), \
                                                                inputs_u_s.to(self.device), inputs_u_s1.to(self.device)
                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3 * self.cfg.DATA.MU_U + 1)

                feat = self.model(inputs)
                feat = self.tuner.learn(feat)
                feat = de_interleave(feat, 3 * self.cfg.DATA.MU_U + 1)

                output = self.tuner.head(feat)
                output_x = output[:batch_size]
                output_u_w, output_u_s, output_u_s1 = output[batch_size:].chunk(3)

                del feat
                del output

                Lx = F.cross_entropy(output_x, targets_x, reduction='mean')

                pseu = torch.softmax(output_u_w.detach(), dim=-1)
                conf, targets_u = torch.max(pseu, dim=-1)
                mask = conf.ge(self.th)
                mask_twice = torch.cat([mask, mask], dim=0).to(self.device)
                output_u_s_twice = torch.cat([output_u_s, output_u_s1], dim=0).to(self.device)
                targets_u_twice = torch.cat([targets_u, targets_u], dim=0).to(self.device)


                if torch.sum(mask_twice) > 0:
                    Lu = (F.cross_entropy(output_u_s_twice, targets_u_twice,
                                          reduction='none') * mask_twice).mean()
                else:
                    Lu = 0


                loss = Lx + Lu

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                current_logit_scale = self.tuner.head.logit_scale.item()
                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())

                batch_time.update(time.time() - end)

                meet_freq = (self.batch_idx + 1) % self.cfg.print_freq == 0
                only_few_batches = self.num_batches < self.cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += self.num_batches - self.batch_idx - 1
                    nb_remain += (
                                         self.num_epochs - self.epoch - 1
                                 ) * self.num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.num_epochs}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"s {current_logit_scale:.4f}"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = self.epoch * self.num_batches + self.batch_idx
                self._writer.add_scalar("train/loss", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/lr", current_lr, n_iter)

                if (self.batch_idx + 1) == self.num_batches:
                    self.sched.step()

                end = time.time()

            last_epoch = (self.epoch + 1) == self.num_epochs
            meet_checkpoint_freq = (
                (self.epoch + 1) % self.cfg.checkpoint_freq == 0
                if self.cfg.checkpoint_freq > 0 else False
            )

            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

            # torch.cuda.empty_cache()

            avg_time_cost = batch_time.avg

            acc_now = self.test_black()
            best_acc = max(best_acc, acc_now)
            self.logger.append([acc_now, best_acc, self.epoch + 1])

        print("Finish training")

        print("Deploy the last-epoch model for testing")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # self.logger.append([f"Elapsed: {elapsed}"]) ## record time
        # Close writer
        self._writer.close()

        self.logger.close()

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
    @torch.no_grad()
    def test_and_save_pseudo_dist(self, save_pt_path="pseudo_stats_test.pt"):
        self.model.eval()
        self.tuner.eval()
        print(f"Evaluate on the test set (and collect pseudo-label distribution)")

        num_classes = self.num_classes  # <- 你按自己工程里的变量名改
        pred_count = torch.zeros(num_classes, dtype=torch.long)  # 统计每类pred数量

        preds = np.array([], dtype=np.int64)
        targets = np.array([], dtype=np.int64)

        for batch in tqdm(self.test_loader, ascii=True):
            image = batch[0].to(self.device, non_blocking=True)
            label = batch[1].to(self.device, non_blocking=True)

            feat = self.model(image)
            output = self.tuner.head(feat)
            del feat

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)  # pred: [B]

            # ---- 统计伪标签分布（按pred计数）----
            # bincount 需要在 CPU 上且指定 minlength
            pred_count += torch.bincount(pred.detach().cpu(), minlength=num_classes)

            # （可选）你原来的 acc 计算保留
            preds = np.append(preds, pred.detach().cpu().numpy().astype(np.int64))
            targets = np.append(targets, label.detach().cpu().numpy().astype(np.int64))

        # ---- acc ----
        acc = float((preds == targets).mean())

        # ---- 分布：count -> 概率 ----
        total = int(pred_count.sum().item())
        pred_dist = pred_count.float() / max(total, 1)  # shape [C], sum=1

        # ---- 排序（降序）----
        sort_idx = torch.argsort(pred_dist, descending=True)
        pred_dist_sorted = pred_dist[sort_idx]
        pred_count_sorted = pred_count[sort_idx]

        # ---- 保存为 pt ----
        pack = {
            "acc": acc,
            "num_classes": num_classes,
            "total_samples": total,
            "pred_count": pred_count,                 # [C]
            "pred_dist": pred_dist,                   # [C]
            "sort_idx": sort_idx,                     # [C]
            "pred_count_sorted": pred_count_sorted,   # [C]
            "pred_dist_sorted": pred_dist_sorted,     # [C]
        }
        torch.save(pack, save_pt_path)
        print(f"[Saved] pseudo stats -> {save_pt_path}")

        return acc

    @torch.no_grad()
    def test_black(self):
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
            feat = self.tuner.learn(feat)
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

    @torch.no_grad()
    def offline_ensemble_test(self):
    
        cfg = self.cfg
        # clip_model1, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='ckpt/metaclip-vit-b16-400m.bin')
        clip_model1, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='ckpt/openclip-vit-b16.bin')

        # clip_model1 = load_clip_to_cpu(cfg)

        clip_model1.half()
        clip_model1.to(self.device)

        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model1.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        
        self.model1 = Model_linear(cfg, clip_model1, self.text_features)
        # self.model1 = Model(cfg, clip_model1, self.text_features)
        self.model1 = self.model1.to(self.device)
        state_dict1 = self.load_model_stat('out/semi_aves/fixmatch_linear_openclip/Semi_Aves/NUML50_imbl1.0_imbu1.0/epoch_30', epoch=30)
        self.model1.tuner.load_state_dict(state_dict1, strict=False)
        self.model1.tuner.eval()

        # # the third model
        # clip_model2, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='ckpt/DFN2B-clip.bin')
        # clip_model2.half()
        # clip_model2.to(self.device)

        # with torch.no_grad():
        #     text_features2 = clip_model2.encode_text(prompts)
        #     text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)

        # self.text_features2 = text_features2
        # self.model2 = Model_linear(cfg, clip_model1, self.text_features2)
        # # self.model2 = Model(cfg, clip_model2, self.text_features2)
        # self.model2 = self.model2.to(self.device)
        # state_dict2 = self.load_model('out/semi_aves/DFN2B-clip/Semi_Aves/NUML50_imbl1.0_imbu1.0/epoch_30', epoch=30)
        # self.model2.tuner.load_state_dict(state_dict2, strict=False)
        # self.model2.tuner.eval()

        state_dict = self.load_model_stat('out/semi_aves/fixmatch_clip/Semi_Aves/NUML50_imbl1.0_imbu1.0/epoch_30', epoch=30)
        self.tuner.load_state_dict(state_dict, strict=False)
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

            feat1 = self.model1(image)
            feat1 = self.model1.tuner.learn(feat1)
            output1 = self.model1.tuner.head(feat1)

            # feat2 = self.model2(image)
            # feat2 = self.model2.tuner.learn(feat2)
            # output2 = self.model2.tuner.head(feat2)


            del feat
            del feat1
            # del feat2

            output = 0.5* output1 + 0.5* output
            # output = 0.3* output1 + 0.4* output + 0.3* output2

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
