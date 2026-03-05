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

        # ===================== CoLA init =====================
        import torch.nn.functional as F

        self.it = 0  # global iteration counter (CoLA 用 it 做 atau schedule) :contentReference[oaicite:1]{index=1}
        self.T = float(getattr(cfg, "T", 1.0))

        # CoLA hyper-params
        self.tau1  = float(getattr(cfg, "tau1", 2.0))
        self.tau12 = float(getattr(cfg, "tau12", 2.0))
        self.tau2  = float(getattr(cfg, "tau2", 2.0))
        self.est_epoch   = int(getattr(cfg, "est_epoch", 5))
        self.est_epoch_2 = int(getattr(cfg, "est_epoch_2", 208))
        self.est_epoch_3 = int(getattr(cfg, "est_epoch_3", 240))
        self.ema_u = float(getattr(cfg, "ema_u", 0.9))

        # CoLA uses p_cutoff (就是 threshold)
        self.p_cutoff = float(getattr(cfg, "th", 0.95))

        # it_per_epoch: CoLA 代码默认 1024；你这里直接用 cfg.total_steps（否则你自己把 1024 写死也行）
        self.it_per_epoch = int(getattr(cfg, "it_per_epoch", cfg.total_steps))
        self.num_eval_iter = int(getattr(cfg, "num_eval_iter", cfg.total_steps))  # CoLA 用它做 count_KL 平均 :contentReference[oaicite:2]{index=2}

        # labeled distribution py_con (prob)
        cnt = torch.zeros(self.num_classes, device=self.device, dtype=torch.float)
        for batch in self.train_label_loader:
            y = batch[1].to(self.device).long()
            cnt += torch.bincount(y, minlength=self.num_classes).float()
        self.py_con = (cnt / cnt.sum().clamp(min=1.0)).detach()                    # CoLA: lb_class_dist/sum :contentReference[oaicite:3]{index=3}
        self.py_uni = (torch.ones(self.num_classes, device=self.device) / self.num_classes)
        self.py_rev = torch.flip(self.py_con, dims=[0])                            # :contentReference[oaicite:4]{index=4}

        # adjustments: l1 changes, l12/l2 fixed (CoLA exactly这样) :contentReference[oaicite:5]{index=5}
        self.adjustment_l1  = torch.log(self.py_con ** self.tau1  + 1e-12)
        self.adjustment_l12 = torch.log(self.py_con ** self.tau12 + 1e-12)
        self.adjustment_l2  = torch.log(self.py_con ** self.tau2  + 1e-12)

        self.taumin = 0.0
        self.taumax = self.tau1                                                    # CoLA: taumax=tau1 :contentReference[oaicite:6]{index=6}

        # KL tracking + unlabeled dist estimate u_py
        self.count_KL = torch.zeros(3, device=self.device)
        self.KL_div = nn.KLDivLoss(reduction="sum")                                # :contentReference[oaicite:7]{index=7}
        self.u_py = (torch.ones(self.num_classes, device=self.device) / self.num_classes)  # :contentReference[oaicite:8]{index=8}

        # atau + optimizer_atau (CoLA exactly这样) :contentReference[oaicite:9]{index=9}
        self.atau = nn.Parameter(torch.tensor(2.0, device=self.device))
        self.optimizer_atau = torch.optim.SGD(
            [self.atau],
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True
        )

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
    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()

    @staticmethod
    def effective_rank(feats: torch.Tensor) -> float:
        if feats.shape[0] == 0:
            return 0.0
        if feats.shape[0] == 1:
            return 1.0
        try:
            feats = feats - feats.mean(dim=0, keepdim=True)
            if torch.allclose(feats, torch.zeros_like(feats), atol=1e-12):
                return 0.0
            U, S, V = torch.svd(feats)
            S = S[S > 1e-12]
            if len(S) == 0:
                return 0.0
            elif len(S) == 1:
                return 1.0
            p = S / S.sum()
            H = -torch.sum(p * torch.log(p + 1e-12))
            eff_rank = torch.exp(H)
            if torch.isnan(eff_rank) or torch.isinf(eff_rank):
                return 0.0
            return eff_rank.item()
        except Exception:
            return 0.0
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
            # ===================== CoLA epoch tau update =====================
            if self.epoch < self.est_epoch:
                self.tau = torch.tensor(self.taumax, device=self.device)
            elif self.epoch >= self.est_epoch and self.epoch < self.est_epoch_3:
                k = self.count_KL / float(self.num_eval_iter)
                KL_softmax = torch.exp(k[0]) / (torch.exp(k[0]) + torch.exp(k[1]) + torch.exp(k[2]) + 1e-12)
                self.tau = self.taumin + (self.taumax - self.taumin) * KL_softmax
                if not torch.isnan(self.tau):
                    self.adjustment_l1 = torch.log(self.py_con ** float(self.tau.item()) + 1e-12)
            else:
                self.tau = self.atau.detach()
                self.adjustment_l1 = self.u_py * self.tau                                 # CoLA 最后阶段就是这样 :contentReference[oaicite:10]{index=10}

            self.count_KL.zero_()  # reset each epoch :contentReference[oaicite:11]{index=11}
            bar = Bar('Training', max=self.cfg.total_steps)
            for self.batch_idx in range(self.cfg.total_steps):
                data_time.update(time.time() - end)

                (inputs_x, targets_x, _) = next(label_loader_iter)
                batch_size = inputs_x.shape[0]
                ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, _) = next(unlabel_loader_iter)

                targets_x = targets_x.to(self.device)
                targets_x = targets_x.to(torch.long)


                inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_x.to(self.device), inputs_u_w.to(self.device), \
                                                                inputs_u_s.to(self.device), inputs_u_s1.to(self.device)
                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3 * self.cfg.DATA.MU_U + 1)

                feat = self.model(inputs)
                feat = de_interleave(feat, 3 * self.cfg.DATA.MU_U + 1)
                feat_x = feat[:batch_size]
                feat_u_w, feat_u_s, feat_u_s1 = feat[batch_size:].chunk(3)
                
                logits   = self.tuner.head(feat)     # main branch
                logits_b = self.tuner.head1(feat)    # aux branch (CoLA 的 logits_aux)

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s, logits_u_s1 = logits[batch_size:].chunk(3)

                logits_x_b = logits_b[:batch_size]
                logits_u_w_b, logits_u_s_b, logits_u_s1_b = logits_b[batch_size:].chunk(3)

                del feat
                del logits
                del logits_b


                # -------- supervised (main + aux) --------
                Lx   = F.cross_entropy(logits_x, targets_x, reduction="mean")
                adj_l2 = self.adjustment_l2.to(self.device, dtype=logits_x_b.dtype).view(1, -1)
                Lx_b = F.cross_entropy(logits_x_b + adj_l2, targets_x, reduction="mean")      # :contentReference[oaicite:12]{index=12}

                # -------- atau schedule (CoLA) --------
                it_epoch = self.it_per_epoch
                if self.it == self.est_epoch_2 * it_epoch:
                    with torch.no_grad():
                        self.atau.data = self.tau.data                                        # :contentReference[oaicite:13]{index=13}

                Lx_atau = None
                if (self.it >= self.est_epoch_2 * it_epoch) and (self.it <= self.est_epoch_3 * it_epoch):
                    ratio = (self.u_py / self.py_con.clamp(min=1e-12))
                    resample_ratio = ratio / ratio.max().clamp(min=1e-12)
                    mask_atau = self.bernouli_mask(resample_ratio[targets_x])                 # :contentReference[oaicite:14]{index=14}

                    adj_atau = (self.u_py * self.atau + 1e-12).to(self.device, dtype=logits_x.dtype).view(1, -1)
                    loss_vec = F.cross_entropy(logits_x.detach() - adj_atau, targets_x, reduction="none")
                    Lx_atau = (loss_vec * mask_atau).mean()

                # -------- pseudo labels (CoLA) --------
                adj_l1  = self.adjustment_l1.to(self.device, dtype=logits_u_w.dtype).view(1, -1)
                adj_l12 = self.adjustment_l12.to(self.device, dtype=logits_u_w.dtype).view(1, -1)

                pseudo_label     = torch.softmax((logits_u_w.detach() - adj_l1)  / self.T, dim=-1)
                pseudo_label_h2  = torch.softmax((logits_u_w.detach() - adj_l12) / self.T, dim=-1)
                pseudo_label_t   = torch.softmax((logits_u_w.detach()) / self.T, dim=-1)
                pseudo_label_b   = torch.softmax((logits_u_w_b.detach()) / self.T, dim=-1)    # aux branch :contentReference[oaicite:15]{index=15}

                max_probs,    targets_u    = torch.max(pseudo_label,    dim=-1)
                max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)
                max_probs_t,  _            = torch.max(pseudo_label_t,  dim=-1)
                max_probs_b,  targets_u_b  = torch.max(pseudo_label_b,  dim=-1)

                mask    = max_probs.ge(self.p_cutoff)
                mask_h2 = max_probs_h2.ge(self.p_cutoff)
                mask_t  = max_probs_t.ge(self.p_cutoff)
                mask_b  = max_probs_b.ge(self.p_cutoff)

                mask_ss_b_h2 = (mask_b + mask_h2).float()
                mask_ss_t    = (mask + mask_t).float()                                       # :contentReference[oaicite:16]{index=16}

                # -------- effective-rank -> update u_py + count_KL (CoLA) --------
                with torch.no_grad():
                    feats_selected = feat_u_w[mask_b.bool()]
                    targets_selected = targets_u_b[mask_b.bool()]
                    feats_by_class = [feats_selected[targets_selected == k] for k in range(self.num_classes)]
                    eff_ranks = [self.effective_rank(f) for f in feats_by_class]              # :contentReference[oaicite:17]{index=17}

                if self.epoch >= self.est_epoch:
                    now_mask = torch.tensor(eff_ranks, device=self.device, dtype=torch.float)
                    if now_mask.sum() > 0:
                        now_mask = now_mask / now_mask.sum()
                        self.u_py = self.ema_u * self.u_py + (1.0 - self.ema_u) * now_mask

                        KL_con = 0.5 * self.KL_div(self.py_con.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_con)
                        KL_uni = 0.5 * self.KL_div(self.py_uni.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_uni)
                        KL_rev = 0.5 * self.KL_div(self.py_rev.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_rev)
                        self.count_KL[0] += KL_con
                        self.count_KL[1] += KL_uni
                        self.count_KL[2] += KL_rev                                                     # :contentReference[oaicite:18]{index=18}

                # -------- unsup losses (用你已有的 2 strong view: logits_u_s + logits_u_s1) --------
                logits_u_s_twice    = torch.cat([logits_u_s,    logits_u_s1],    dim=0)
                logits_u_s_b_twice  = torch.cat([logits_u_s_b,  logits_u_s1_b],  dim=0)
                targets_u_twice     = torch.cat([targets_u,     targets_u],      dim=0)
                targets_u_h2_twice  = torch.cat([targets_u_h2,  targets_u_h2],   dim=0)
                mask_twice_ss_t     = torch.cat([mask_ss_t,     mask_ss_t],      dim=0)
                mask_twice_ss_b_h2  = torch.cat([mask_ss_b_h2,  mask_ss_b_h2],   dim=0)

                Lu   = (F.cross_entropy(logits_u_s_twice,   targets_u_twice,    reduction="none") * mask_twice_ss_t).mean()
                Lu_b = (F.cross_entropy(logits_u_s_b_twice, targets_u_h2_twice, reduction="none") * mask_twice_ss_b_h2).mean()  # :contentReference[oaicite:19]{index=19}

                loss = Lx + Lx_b + Lu + Lu_b

                if mask.any():
                    y = targets_u[mask].detach().cpu()
                    pseudo_count += torch.bincount(y, minlength=self.num_classes)
                    n_accept += int(mask.sum().item())


                                # ---- main optimizer (keep your scaler if you have) ----
                self.optim.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()

                # ---- atau optimizer (only updates atau; logits_x is detached so no graph conflict) ----
                if Lx_atau is not None and self.atau.requires_grad:
                    self.optimizer_atau.zero_grad()
                    Lx_atau.backward()
                    self.optimizer_atau.step()

                # ---- disable atau grad exactly like CoLA ---- :contentReference[oaicite:20]{index=20}
                if self.it == self.est_epoch_3 * self.it_per_epoch + 1:
                    self.atau.requires_grad = False

                self.it += 1

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
