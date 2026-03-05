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
        # ================= Meta-Expert hyperparams / states =================
        self.cut1 = int(getattr(cfg, "cut1", self.num_classes // 3))
        self.cut2 = int(getattr(cfg, "cut2", 2 * self.num_classes // 3))
        assert 0 < self.cut1 < self.cut2 < self.num_classes

        self.beta1 = float(getattr(cfg, "beta1", 0.9999))
        self.beta2 = float(getattr(cfg, "beta2", 0.9999))

        # logit-adjustment strengths (can be float or list/np.array; you can decide)
        self.tau_lb1 = getattr(cfg, "la_tau_lb1", 0.0)
        self.tau_lb2 = getattr(cfg, "la_tau_lb2", 0.0)
        self.tau_lb3 = getattr(cfg, "la_tau_lb3", 0.0)

        # use your fixmatch weight as lambda_u
        self.lambda_u = float(getattr(cfg, "w_con", 1.0))
        self.p_cutoff = float(getattr(cfg, "th", 0.95))  # Meta-Expert uses p_cutoff

        # labeled class histogram (counts + p_hat_lb)
        cnt = torch.zeros(self.num_classes, device=self.device, dtype=torch.float)
        for batch in self.train_label_loader:
            y = batch[1].to(self.device).long()
            cnt += torch.bincount(y, minlength=self.num_classes).float()

        self.p_hat_lb = (cnt / torch.clamp(cnt.sum(), min=1.0)).detach()  # [C]
        self.log_p_hat_lb = torch.log(self.p_hat_lb + 1e-12)              # [C]

        # effective-number weights (exactly same formula pattern)
        def _effnum_weight(beta: float, n: torch.Tensor):
            # n: [K] counts (float)
            w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=n.device), n).clamp(min=1e-12))
            w = w / w.sum().clamp(min=1e-12) * float(w.numel())
            return w

        # weight1: per-class weights for fuse supervised class CE
        self.weight1 = _effnum_weight(self.beta1, cnt).detach()  # [C]

        # weight2: 3-group weights for fuse supervised H/M/T-membership CE
        cnt_h = cnt[:self.cut1].sum()
        cnt_m = cnt[self.cut1:self.cut2].sum()
        cnt_t = cnt[self.cut2:].sum()
        cnt_hmt = torch.stack([cnt_h, cnt_m, cnt_t]).detach()
        self.weight2 = _effnum_weight(self.beta2, cnt_hmt).detach()  # [3]

        # dynamic expert selection weights (weight_kl) and estimation buffer
        self.est_epoch = int(getattr(cfg, "est_epoch", 5))
        self.current_epoch = self.est_epoch
        self.current_mask = torch.zeros(self.num_classes, device=self.device)
        self.est_class_dist = (torch.ones(self.num_classes, device=self.device) / self.num_classes)
        self.weight_kl = (torch.ones(3, device=self.device) / 3.0)  # [3]
        # ---- make CE weights dtype consistent with model (fp16/fp32) ----
        model_dtype = next(self.tuner.parameters()).dtype
        self.weight1 = self.weight1.to(device=self.device, dtype=model_dtype)
        self.weight2 = self.weight2.to(device=self.device, dtype=model_dtype)
        
    def _metaexpert_epoch_hook(self):
        """
        Exactly follow Meta-Expert's per-epoch update:
        after est_epoch, at the first iter of each epoch, update current_epoch,
        and at epoch == est_epoch+2, compute weight_kl from last epoch's current_mask.
        """
        if (self.epoch > self.est_epoch) and (self.epoch > self.current_epoch):
            self.current_epoch += 1
            if self.epoch == self.est_epoch + 2:
                if self.current_mask.sum() > 0:
                    self.est_class_dist = self.current_mask / self.current_mask.sum().clamp(min=1e-12)

                    C2 = int(self.num_classes / 2)
                    tail_head_ratio = (self.est_class_dist[C2:].max() / self.est_class_dist[:C2].min().clamp(min=1e-12))
                    head_tail_ratio = (self.est_class_dist[:C2].max() / self.est_class_dist[C2:].min().clamp(min=1e-12))

                    self.weight_kl[2] = ((tail_head_ratio > 2.0) & (head_tail_ratio < 2.0)).float()
                    self.weight_kl[1] = ((self.est_class_dist.max() / self.est_class_dist.min().clamp(min=1e-12)) < 2.0).float()
                    self.weight_kl[0] = 1.0 - ((self.weight_kl[2] + self.weight_kl[1]).eq(1.0)).float()

                self.current_mask.zero_()

    def _metaexpert_loss(self, outputs: dict, targets_x: torch.Tensor, batch_size: int):
        """
        Full Meta-Expert loss, adapted to your 4-part batch:
        [lb | ulb_w | ulb_s | ulb_s1]. We *only* use ulb_s to match Meta-Expert.
        """
        num_lb = batch_size
        lb = targets_x  # no noise branch in your code

        # ---- split helper: return (lb, ulb_w, ulb_s, ulb_s1) ----
        def _split3(t):
            lb_t = t[:num_lb]
            ulb_w, ulb_s = t[num_lb:].chunk(2)
            return lb_t, ulb_w, ulb_s

        # head logits
        logits_x_lb1, logits_x_ulb_w1, logits_x_ulb_s1 = _split3(outputs["logits"])
        logits_x_lb2, logits_x_ulb_w2, logits_x_ulb_s2 = _split3(outputs["aux_logits1"])
        logits_x_lb3, logits_x_ulb_w3, logits_x_ulb_s3 = _split3(outputs["aux_logits2"])

        # experts (只取 ulb_s 部分)
        _, _, logits_x_ulb_sH1 = _split3(outputs["logitsH"])
        _, _, logits_x_ulb_sM1 = _split3(outputs["logitsM"])
        _, _, logits_x_ulb_sT1 = _split3(outputs["logitsT"])

        _, _, logits_x_ulb_sH2 = _split3(outputs['aux_logitsH1'])
        _, _, logits_x_ulb_sM2 = _split3(outputs['aux_logitsM1'])
        _, _, logits_x_ulb_sT2 = _split3(outputs['aux_logitsT1'])

        _, _, logits_x_ulb_sH3 = _split3(outputs['aux_logitsH2'])
        _, _, logits_x_ulb_sM3 = _split3(outputs['aux_logitsM2'])
        _, _, logits_x_ulb_sT3 = _split3(outputs['aux_logitsT2'])

        # ====== logit adjustment for supervised (tau * log p_hat_lb) ======
        tau1 = torch.tensor(self.tau_lb1, device=self.device, dtype=logits_x_lb1.dtype)
        tau2 = torch.tensor(self.tau_lb2, device=self.device, dtype=logits_x_lb2.dtype)
        tau3 = torch.tensor(self.tau_lb3, device=self.device, dtype=logits_x_lb3.dtype)

        log_p = self.log_p_hat_lb.to(device=logits_x_lb1.device, dtype=logits_x_lb1.dtype)
        sup_loss1 = F.cross_entropy(logits_x_lb1 + tau1 * log_p, lb, reduction='mean')
        sup_loss2 = F.cross_entropy(logits_x_lb2 + tau2 * log_p, lb, reduction='mean')
        sup_loss3 = F.cross_entropy(logits_x_lb3 + tau3 * log_p, lb, reduction='mean')

        # ====== pseudo labels (FixMatch style) ======
        probs_x_ulb_w1 = torch.softmax(logits_x_ulb_w1.detach(), dim=-1)
        mask1 = probs_x_ulb_w1.amax(dim=-1).ge(self.p_cutoff)     # [B_u]
        pseudo_label1 = probs_x_ulb_w1.argmax(dim=-1)             # [B_u]

        probs_x_ulb_w2 = torch.softmax(logits_x_ulb_w2.detach(), dim=-1)
        mask2 = probs_x_ulb_w2.amax(dim=-1).ge(self.p_cutoff)
        pseudo_label2 = probs_x_ulb_w2.argmax(dim=-1)

        probs_x_ulb_w3 = torch.softmax(logits_x_ulb_w3.detach(), dim=-1)
        mask3 = probs_x_ulb_w3.amax(dim=-1).ge(self.p_cutoff)
        pseudo_label3 = probs_x_ulb_w3.argmax(dim=-1)

        m1 = mask1.float()
        m2 = mask2.float()
        m3 = mask3.float()

        # nested weights: H = all accepted, M = accepted & (>=cut1), T = accepted & (>=cut2)
        pseudo_label1H = m1
        pseudo_label1M = (pseudo_label1 >= self.cut1).float() * m1
        pseudo_label1T = (pseudo_label1 >= self.cut2).float() * m1

        pseudo_label2H = m2
        pseudo_label2M = (pseudo_label2 >= self.cut1).float() * m2
        pseudo_label2T = (pseudo_label2 >= self.cut2).float() * m2

        pseudo_label3H = m3
        pseudo_label3M = (pseudo_label3 >= self.cut1).float() * m3
        pseudo_label3T = (pseudo_label3 >= self.cut2).float() * m3

        # ====== unsup losses: CE on H/M/T logits with those weights ======
        def _unsup_loss(logitH, logitM, logitT, y, wH, wM, wT):
            loss = (F.cross_entropy(logitH, y, reduction='none') * wH).sum()
            loss = loss + (F.cross_entropy(logitM, y, reduction='none') * wM).sum()
            loss = loss + (F.cross_entropy(logitT, y, reduction='none') * wT).sum()
            denom = (wH.sum() + wM.sum() + wT.sum() + 1e-12)
            return loss / denom

        unsup_loss1 = _unsup_loss(logits_x_ulb_sH1, logits_x_ulb_sM1, logits_x_ulb_sT1,
                                pseudo_label1, pseudo_label1H, pseudo_label1M, pseudo_label1T)
        unsup_loss2 = _unsup_loss(logits_x_ulb_sH2, logits_x_ulb_sM2, logits_x_ulb_sT2,
                                pseudo_label2, pseudo_label2H, pseudo_label2M, pseudo_label2T)
        unsup_loss3 = _unsup_loss(logits_x_ulb_sH3, logits_x_ulb_sM3, logits_x_ulb_sT3,
                                pseudo_label3, pseudo_label3H, pseudo_label3M, pseudo_label3T)

        # ====== estimate unlabeled class dist (exact behavior: update current_mask using head-2 pseudo labels) ======
        if (self.epoch > self.est_epoch) and (self.epoch == self.current_epoch):
            self.current_mask.scatter_add_(0, pseudo_label2, mask2.float())

        # ===================== fuse losses =====================
        # split fuse outputs
        fuse_l_logits_x_lb, fuse_l_logits_x_ulb_w, fuse_l_logits_x_ulb_s  = _split3(outputs['fuse_logit_l'])

        _, fuse_c_logits_HMT_x_ulb_w_1, fuse_c_logits_HMT_x_ulb_s_1 = _split3(outputs['fuse_logit_HMT_c_1'])
        _, fuse_c_logits_HMT_x_ulb_w_2, fuse_c_logits_HMT_x_ulb_s_2 = _split3(outputs['fuse_logit_HMT_c_2'])
        _, fuse_c_logits_HMT_x_ulb_w_3, fuse_c_logits_HMT_x_ulb_s_3 = _split3(outputs['fuse_logit_HMT_c_3'])

        fuse_logits_w_lb, fuse_logits_w_ulb_w, fuse_logits_w_ulb_s = _split3(outputs['fuse_w_logit'])

        # supervised fuse (class)
        sup_fuse_loss1 = F.cross_entropy(fuse_l_logits_x_lb, lb, weight=self.weight1)

        # unsupervised fuse (choose which expert via weight_kl)
        fuse_probs_x_ulb_w_1 = torch.softmax(fuse_c_logits_HMT_x_ulb_w_1.detach(), dim=-1)
        fuse_pseudo_label11 = fuse_probs_x_ulb_w_1.argmax(dim=-1)

        fuse_probs_x_ulb_w_2 = torch.softmax(fuse_c_logits_HMT_x_ulb_w_2.detach(), dim=-1)
        fuse_pseudo_label12 = fuse_probs_x_ulb_w_2.argmax(dim=-1)

        fuse_probs_x_ulb_w_3 = torch.softmax(fuse_c_logits_HMT_x_ulb_w_3.detach(), dim=-1)
        fuse_pseudo_label13 = fuse_probs_x_ulb_w_3.argmax(dim=-1)

        unsup_fuse_loss1 = self.weight_kl[0] * F.cross_entropy(fuse_c_logits_HMT_x_ulb_s_1, fuse_pseudo_label11)
        unsup_fuse_loss1 = unsup_fuse_loss1 + self.weight_kl[1] * F.cross_entropy(fuse_c_logits_HMT_x_ulb_s_2, fuse_pseudo_label12)
        unsup_fuse_loss1 = unsup_fuse_loss1 + self.weight_kl[2] * F.cross_entropy(fuse_c_logits_HMT_x_ulb_s_3, fuse_pseudo_label13)

        # supervised fuse (predict H/M/T membership)
        lb_w1 = torch.where(lb < self.cut1, torch.ones_like(lb), torch.zeros_like(lb))
        lb_w2 = torch.where((self.cut1 <= lb) & (lb < self.cut2), torch.ones_like(lb), torch.zeros_like(lb))
        lb_w3 = torch.where(self.cut2 <= lb, torch.ones_like(lb), torch.zeros_like(lb))
        lb_w = 0 * lb_w1 + 1 * lb_w2 + 2 * lb_w3

        sup_fuse_loss2 = F.cross_entropy(fuse_logits_w_lb, lb_w, weight=self.weight2)

        # unsupervised fuse (membership)
        fuse_probs_w_ulb_w = torch.softmax(fuse_logits_w_ulb_w.detach(), dim=-1)
        fuse_pseudo_label2 = fuse_probs_w_ulb_w.argmax(dim=-1)
        unsup_fuse_loss2 = F.cross_entropy(fuse_logits_w_ulb_s, fuse_pseudo_label2)

        sup_fuse_loss = sup_fuse_loss1 + sup_fuse_loss2
        unsup_fuse_loss = unsup_fuse_loss1 + unsup_fuse_loss2

        # warmup: disable fuse losses for early epochs (exactly as code)
        if self.epoch <= self.est_epoch + 1:
            sup_fuse_loss = torch.zeros([], device=self.device, dtype=sup_fuse_loss.dtype)
            unsup_fuse_loss = torch.zeros([], device=self.device, dtype=unsup_fuse_loss.dtype)

        # total
        sup_loss = sup_loss1 + sup_loss2 + sup_loss3 + sup_fuse_loss
        unsup_loss = self.lambda_u * unsup_loss1 + self.lambda_u * unsup_loss2 + self.lambda_u * unsup_loss3 + self.lambda_u * unsup_fuse_loss
        total_loss = sup_loss + unsup_loss

        # for logging
        util_ratio = mask2.float().mean().item()
        return total_loss, util_ratio, pseudo_label2, mask2   
    
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
            self._metaexpert_epoch_hook()
            bar = Bar('Training', max=self.cfg.total_steps)
            for self.batch_idx in range(self.cfg.total_steps):
                data_time.update(time.time() - end)

                (inputs_x, targets_x, _) = next(label_loader_iter)
                batch_size = inputs_x.shape[0]
                ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, _) = next(unlabel_loader_iter)
                inputs_x = inputs_x.to(self.device)
                inputs_u_w = inputs_u_w.to(self.device)
                inputs_u_s = inputs_u_s.to(self.device)
                targets_x = targets_x.to(self.device)
                targets_x = targets_x.to(torch.long)

                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)),   # 注意：不拼 inputs_u_s1
                    2 * self.cfg.DATA.MU_U + 1
                )

                layers = getattr(self.cfg, "meta_layers", [2,5,8,11])
                feat, feats_dict = self.model(inputs, return_layers=layers)

                size = 2 * self.cfg.DATA.MU_U + 1
                feat  = de_interleave(feat, size)
                feat1 = de_interleave(feats_dict[layers[0]], size)
                feat2 = de_interleave(feats_dict[layers[1]], size)
                feat3 = de_interleave(feats_dict[layers[2]], size)
                feat4 = de_interleave(feats_dict[layers[3]], size)

                feat_for_fuse = {"feat1": feat1, "feat2": feat2, "feat3": feat3, "feat4": feat4}

                outputs = self.tuner.forward_metaexpert(
                    feat, feat_for_fuse,
                    p_hat_lb=self.p_hat_lb,
                    tau1=self.tau_lb1, tau2=self.tau_lb2, tau3=self.tau_lb3
                )

                loss, util_ratio, pseudo_label2, mask2 = self._metaexpert_loss(outputs, targets_x, batch_size)
                if mask2.any():
                    y = pseudo_label2[mask2].detach().cpu()
                    pseudo_count += torch.bincount(y, minlength=self.num_classes)
                    n_accept += int(mask2.sum().item())
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
        layers = getattr(self.cfg, "meta_layers", [2,5,8,11])
        for batch in tqdm(self.test_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            feat, feats_dict = self.model(image, return_layers=layers)
            # de_interleave 不需要（test 没 interleave）
            feat_for_fuse = {"feat1": feats_dict[layers[0]], "feat2": feats_dict[layers[1]],
                            "feat3": feats_dict[layers[2]], "feat4": feats_dict[layers[3]]}

            outputs = self.tuner.forward_metaexpert(feat, feat_for_fuse, self.p_hat_lb, self.tau_lb1, self.tau_lb2, self.tau_lb3)

            pred = outputs["fuse_logit_l"].argmax(dim=1)

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
