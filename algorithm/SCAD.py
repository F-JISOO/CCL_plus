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
import math

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

def _create_coarse_labels(superclass_indices_list, num_classes: int):
    coarse = torch.zeros(num_classes, dtype=torch.long)
    for k, idxs in enumerate(superclass_indices_list):
        coarse[torch.tensor(idxs, dtype=torch.long)] = k
    return coarse  # [C] each fine class -> superclass id

def _coarse_logits_from_fine_logits(fine_logits: torch.Tensor, superclass_indices_list):
    # fine_logits: [B, C] -> coarse_logits: [B, K]
    chunks = []
    for idxs in superclass_indices_list:
        idxs_t = torch.tensor(idxs, device=fine_logits.device, dtype=torch.long)
        chunks.append(torch.logsumexp(fine_logits.index_select(1, idxs_t), dim=1))
    return torch.stack(chunks, dim=1)

def _build_intra_delta(superclass_indices_list, py_con: torch.Tensor):
    # returns delta_kc: [K, C]  (matches SCAD build_intra_delta behavior)
    # py_con can be prob or counts; only relative ratios matter
    device = py_con.device
    C = int(py_con.numel())
    K = len(superclass_indices_list)
    delta = torch.zeros((K, C), device=device, dtype=torch.float)

    # beta_kc = n_kc / max_n_in_superclass
    beta_max = 0.0
    betas = []
    for k, idxs in enumerate(superclass_indices_list):
        idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
        n = py_con.index_select(0, idxs_t).float()
        max_n = torch.clamp(n.max(), min=1e-12)
        beta = n / max_n
        betas.append((idxs_t, beta))
        beta_max = max(beta_max, float(beta.max().item()))

    for k, (idxs_t, beta) in enumerate(betas):
        delta[k, :] = beta_max
        delta[k, idxs_t] = beta
    return delta  # [K, C]

def _sym_kl(p: torch.Tensor, q: torch.Tensor, kl: nn.KLDivLoss):
    # p,q are probs (sum=1), return 0.5*KL(p||q)+0.5*KL(q||p) (SCAD uses this)
    eps = 1e-12
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return 0.5 * kl(p.log(), q) + 0.5 * kl(q.log(), p)

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
        # ---- SCAD: number of super-classes K ----
        if getattr(cfg, "superclass_indices_list", None) is not None:
            cfg.NUM_SUPER = len(cfg.superclass_indices_list)
        else:
            cfg.NUM_SUPER = int(math.ceil(cfg.DATA.NUMBER_CLASSES / 4.0))

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
        # ===== SCAD hyperparams (defaults align with acrwithSCAD/fixmatch_scad) =====
        self.T = float(getattr(cfg, "T", 1.0))
        self.tau1  = float(getattr(cfg, "tau1", 2.0))
        self.tau12 = float(getattr(cfg, "tau12", 2.0))
        self.tau2  = float(getattr(cfg, "tau2", 2.0))
        self.ema_u = float(getattr(cfg, "ema_u", 0.99))
        self.est_epoch = int(getattr(cfg, "est_epoch", 5))
        self.taumin = 0.0
        self.taumax = self.tau2

        # ===== superclass mapping =====
        # 你也可以直接在 cfg.superclass_indices_list 里传入
        name = str(self.cfg.DATA.NAME).lower()
        if hasattr(cfg, "superclass_indices_list"):
            self.superclass_indices_list = cfg.superclass_indices_list
        else:
            # --------- Auto super-class generation (SCAD-style) ---------
            # K = ceil(C/4)  (SCAD Appendix A.9)
            C = self.num_classes
            K = int(getattr(cfg, "NUM_SUPER", math.ceil(C / 4.0)))

            # 用你已经算好的 CLIP text embedding（build_model/build_linear_black_model 里算的 self.text_features）
            # text_features: [C, D]
            text_feat = self.text_features.detach().float().cpu().numpy()

            from sklearn.cluster import AgglomerativeClustering
            try:
                # sklearn 新版：metric
                clusterer = AgglomerativeClustering(n_clusters=K, linkage="average", metric="cosine")
            except TypeError:
                # sklearn 旧版：affinity
                clusterer = AgglomerativeClustering(n_clusters=K, linkage="average", affinity="cosine")

            cluster_id = clusterer.fit_predict(text_feat)  # [C]

            self.superclass_indices_list = [
                np.where(cluster_id == k)[0].tolist() for k in range(K)
                if np.any(cluster_id == k)
            ]

        self.num_super = len(self.superclass_indices_list)
        self.coarse_labels_list = _create_coarse_labels(self.superclass_indices_list, self.num_classes).to(self.device)

        # ===== compute py_con / py_con1 from labeled loader =====
        cnt = torch.zeros(self.num_classes, device=self.device, dtype=torch.float)
        for batch in self.train_label_loader:
            y = batch[1].to(self.device).long()
            cnt += torch.bincount(y, minlength=self.num_classes).float()
        self.py_con = (cnt / torch.clamp(cnt.sum(), min=1.0)).detach()

        cnt_super = torch.zeros(self.num_super, device=self.device, dtype=torch.float)
        for c in range(self.num_classes):
            cnt_super[self.coarse_labels_list[c]] += cnt[c]
        self.py_con1 = (cnt_super / torch.clamp(cnt_super.sum(), min=1.0)).detach()

        self.py_uni  = (torch.ones(self.num_classes, device=self.device) / self.num_classes)
        self.py_rev  = torch.flip(self.py_con, dims=[0])
        self.py_uni1 = (torch.ones(self.num_super, device=self.device) / self.num_super)
        self.py_rev1 = torch.flip(self.py_con1, dims=[0])

        # ===== adjustments =====
        self.adjustment_l1  = compute_adjustment_by_py(self.py_con,  self.tau1,  self.device)
        self.adjustment_l12 = compute_adjustment_by_py(self.py_con,  self.tau12, self.device)
        self.adjustment_l2  = compute_adjustment_by_py(self.py_con,  self.tau2,  self.device)
        self.adjustment_sl1  = compute_adjustment_by_py(self.py_con1, self.tau1,  self.device)
        self.adjustment_sl12 = compute_adjustment_by_py(self.py_con1, self.tau12, self.device)
        self.adjustment_sl2  = compute_adjustment_by_py(self.py_con1, self.tau2,  self.device)

        # ===== unlabeled distribution estimation states =====
        self.u_py  = (torch.ones(self.num_classes, device=self.device) / self.num_classes)
        self.u_py1 = (torch.ones(self.num_super,  device=self.device) / self.num_super)
        self.count_KL  = torch.zeros(3, device=self.device)
        self.count_KL1 = torch.zeros(3, device=self.device)
        self.kl_div  = nn.KLDivLoss(reduction="sum")
        self.est_step = 0

        self.delta_kc_t = _build_intra_delta(self.superclass_indices_list, self.py_con).to(self.device)
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
            # ===== epoch-start: dynamic tau update (use LAST epoch count_KL) =====
            if self.epoch > self.est_epoch:
                k  = self.count_KL  / float(self.cfg.total_steps)
                k1 = self.count_KL1 / float(self.cfg.total_steps)

                KL_softmax  = torch.exp(k[0])  / (torch.exp(k[0])  + torch.exp(k[1])  + torch.exp(k[2])  + 1e-12)
                KL_softmax1 = torch.exp(k1[0]) / (torch.exp(k1[0]) + torch.exp(k1[1]) + torch.exp(k1[2]) + 1e-12)

                tau  = self.taumin + (self.taumax - self.taumin) * KL_softmax
                tau1 = self.taumin + (self.taumax - self.taumin) * KL_softmax1

                if not math.isnan(float(tau.item())):
                    self.adjustment_l1  = compute_adjustment_by_py(self.py_con,  float(tau.item()),  self.device)
                    self.adjustment_sl1 = compute_adjustment_by_py(self.py_con1, float(tau1.item()), self.device)

            # ===== reset counters for THIS epoch (start accumulating again) =====
            self.count_KL.zero_()
            self.count_KL1.zero_()
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

                logits    = self.tuner.head(feat)     # fine
                logits_b  = self.tuner.head1(feat)    # balanced fine (你已有)
                logits_s  = self.tuner.heads(feat)   # coarse
                logits_sb = self.tuner.heads1(feat)  # balanced coarse

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s, logits_u_s1 = logits[batch_size:].chunk(3)

                logits_x_b = logits_b[:batch_size]
                logits_u_w_b, logits_u_s_b, logits_u_s1_b = logits_b[batch_size:].chunk(3)

                logits_sx = logits_s[:batch_size]
                logits_u_sw, logits_u_ss, logits_u_ss1 = logits_s[batch_size:].chunk(3)

                logits_x_sb = logits_sb[:batch_size]
                logits_u_w_sb, logits_u_s_sb, logits_u_s1_sb = logits_sb[batch_size:].chunk(3)

                del feat
                del logits
                del logits_b
                del logits_s
                del logits_sb

               

                


                                # ===================== SCAD loss (full logic) =====================
                # u_real gate (for unlabeled distribution estimation)
                u_real = u_real.to(self.device) if torch.is_tensor(u_real) else None
                mask_l = (u_real != -2).float() if u_real is not None else None  # [B_u]

                # ---- coarse labels for supervised ----
                targets_sx = self.coarse_labels_list[targets_x]  # [B]

                # ---- supervised losses (fine + coarse) ----
                Lx  = F.cross_entropy(logits_x,  targets_x,  reduction="mean")
                Lx1 = F.cross_entropy(logits_sx, targets_sx, reduction="mean")

                # ---- balanced supervised losses (use balanced heads + adjustment_l2/sl2) ----
                adj_l2  = self.adjustment_l2.to(self.device, dtype=logits_x_b.dtype)
                adj_sl2 = self.adjustment_sl2.to(self.device, dtype=logits_x_sb.dtype)
                Lx_b  = F.cross_entropy(logits_x_b  + adj_l2,  targets_x,  reduction="mean")
                Lx_b1 = F.cross_entropy(logits_x_sb + adj_sl2, targets_sx, reduction="mean")

                # ---- pseudo labels (fine / coarse; adjusted + h2; plus balanced heads) ----
                adj_l1   = self.adjustment_l1.to(self.device, dtype=logits_u_w.dtype)
                adj_l12  = self.adjustment_l12.to(self.device, dtype=logits_u_w.dtype)
                adj_sl1  = self.adjustment_sl1.to(self.device, dtype=logits_u_sw.dtype)
                adj_sl12 = self.adjustment_sl12.to(self.device, dtype=logits_u_sw.dtype)

                # fine pseudos
                pseudo_label     = torch.softmax((logits_u_w.detach() - adj_l1)  / self.T, dim=-1)
                pseudo_label_h2  = torch.softmax((logits_u_w.detach() - adj_l12) / self.T, dim=-1)
                pseudo_label_t   = torch.softmax( logits_u_w.detach() / self.T, dim=-1)
                # balanced fine pseudo MUST come from balanced fine head
                pseudo_label_b   = torch.softmax( logits_u_w_b.detach() / self.T, dim=-1)

                # coarse pseudos
                pseudo_label1    = torch.softmax((logits_u_sw.detach() - adj_sl1)  / self.T, dim=-1)
                pseudo_label_h21 = torch.softmax((logits_u_sw.detach() - adj_sl12) / self.T, dim=-1)
                pseudo_label_st  = torch.softmax( logits_u_sw.detach() / self.T, dim=-1)
                # balanced coarse pseudo MUST come from balanced coarse head
                pseudo_label_sb  = torch.softmax( logits_u_w_sb.detach() / self.T, dim=-1)

                max_probs,    targets_u      = torch.max(pseudo_label,     dim=-1)
                max_probs_h2, targets_u_h2   = torch.max(pseudo_label_h2,  dim=-1)
                max_probs_t,  _              = torch.max(pseudo_label_t,   dim=-1)
                max_probs_b,  targets_u_b    = torch.max(pseudo_label_b,   dim=-1)

                max_probs1,   targets_su     = torch.max(pseudo_label1,    dim=-1)
                max_probs_h21,targets_u_sh2  = torch.max(pseudo_label_h21, dim=-1)
                max_probs_t1, _              = torch.max(pseudo_label_st,  dim=-1)
                max_probs_b1, targets_u_sb   = torch.max(pseudo_label_sb,  dim=-1)

                # ---- penalty_intra + refine pseudo_label_h2 ----
                # (建议：你可以在 __init__ 里缓存 self.delta_kc_t，避免每步重算)
                delta_kc_t = self.delta_kc_t
                delta_kc_t = delta_kc_t.to(self.device, dtype=pseudo_label_h21.dtype)

                penalty_intra = pseudo_label_h21 @ delta_kc_t  # [B_u, C]
                pseudo_label_h2 = torch.softmax(
                    (logits_u_w.detach() - adj_l12 - self.tau12 * penalty_intra) / self.T, dim=-1
                )
                max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)

                # ---- condition trick ----
                condition = (
                    (torch.logical_or(max_probs_h21 >= self.th, max_probs_b1 >= self.th)) &
                    (~torch.logical_or(max_probs_h2  >= self.th, max_probs_b  >= self.th)) &
                    (self.coarse_labels_list[targets_u_h2] == targets_u_sh2)
                )
                max_probs_h2 = max_probs_h2.clone()
                max_probs_h2[condition] = self.th

                # ---- masks (IMPORTANT: add masks => weights {0,1,2}) ----
                mask     = max_probs.ge(self.th)
                mask_t   = max_probs_t.ge(self.th)
                mask_h2  = max_probs_h2.ge(self.th)
                mask_b   = max_probs_b.ge(self.th)

                mask1    = max_probs1.ge(self.th)
                mask_t1  = max_probs_t1.ge(self.th)
                mask_h21 = max_probs_h21.ge(self.th)
                mask_b1  = max_probs_b1.ge(self.th)

                mask_ss_t       = (mask   + mask_t ).float()     # [B_u]
                mask_ss_b_h2    = (mask_b + mask_h2).float()     # [B_u]
                mask_ss_t1      = (mask1  + mask_t1).float()     # [B_u]
                mask_ss_b_h21   = (mask_b1+ mask_h21).float()    # [B_u]

                mask_twice_ss_t      = torch.cat([mask_ss_t,      mask_ss_t],      dim=0)
                mask_twice_ss_b_h2   = torch.cat([mask_ss_b_h2,   mask_ss_b_h2],   dim=0)
                mask_twice_ss_t1     = torch.cat([mask_ss_t1,     mask_ss_t1],     dim=0)
                mask_twice_ss_b_h21  = torch.cat([mask_ss_b_h21,  mask_ss_b_h21],  dim=0)

                # ---- logits/targets twice (strong + strong1) ----
                logits_u_s_twice      = torch.cat([logits_u_s,      logits_u_s1],      dim=0)
                logits_u_s_b_twice    = torch.cat([logits_u_s_b,    logits_u_s1_b],    dim=0)
                logits_u_ss_twice     = torch.cat([logits_u_ss,     logits_u_ss1],     dim=0)
                logits_u_s_sb_twice   = torch.cat([logits_u_s_sb,   logits_u_s1_sb],   dim=0)

                targets_u_twice       = torch.cat([targets_u,      targets_u],      dim=0)
                targets_u_h2_twice    = torch.cat([targets_u_h2,   targets_u_h2],   dim=0)
                targets_su_twice      = torch.cat([targets_su,     targets_su],     dim=0)
                targets_u_sh2_twice   = torch.cat([targets_u_sh2,  targets_u_sh2],  dim=0)

                # ---- unsupervised losses (4 branches) ----
                Lu    = (F.cross_entropy(logits_u_s_twice,     targets_u_twice,      reduction="none") * mask_twice_ss_t).mean()
                Lu_b  = (F.cross_entropy(logits_u_s_b_twice,   targets_u_h2_twice,   reduction="none") * mask_twice_ss_b_h2).mean()
                Lu1   = (F.cross_entropy(logits_u_ss_twice,    targets_su_twice,     reduction="none") * mask_twice_ss_t1).mean()
                Lu_b1 = (F.cross_entropy(logits_u_s_sb_twice,  targets_u_sh2_twice,  reduction="none") * mask_twice_ss_b_h21).mean()

                loss = Lx + Lu + Lx_b + Lu_b + Lx1 + Lu1 + Lx_b1 + Lu_b1

                # ---- stats for pseudo-label acceptance (keep your original bookkeeping) ----
                if mask.any():
                    y = targets_u[mask].detach().cpu()
                    pseudo_count += torch.bincount(y, minlength=self.num_classes)
                    n_accept += int(mask.sum().item())

                # ---- unlabeled distribution estimation + KL accumulation ----
                if (self.epoch > self.est_epoch) and (mask_l is not None):
                    now_mask  = torch.zeros(self.num_classes, device=self.device, dtype=torch.float)
                    now_mask1 = torch.zeros(self.num_super,  device=self.device, dtype=torch.float)

                    w_cls = (mask_l * mask_b.float())          # [B_u]
                    now_mask.scatter_add_(0, targets_u_b, w_cls)

                    w_sup = (mask_l * mask_b1.float())         # [B_u]
                    now_mask1.scatter_add_(0, targets_u_sb, w_sup)

                    eps = 1e-12
                    if now_mask.sum() > 0:
                        now_mask = now_mask / (now_mask.sum() + eps)
                        self.u_py = self.ema_u * self.u_py + (1 - self.ema_u) * now_mask

                        # symmetric KL
                        def _sym_kl_local(p, q):
                            p = torch.clamp(p, min=eps)
                            q = torch.clamp(q, min=eps)
                            return 0.5 * self.kl_div(p.log(), q) + 0.5 * self.kl_div(q.log(), p)

                        self.count_KL[0] += _sym_kl_local(self.py_con, self.u_py)
                        self.count_KL[1] += _sym_kl_local(self.py_uni, self.u_py)
                        self.count_KL[2] += _sym_kl_local(self.py_rev, self.u_py)

                    if now_mask1.sum() > 0:
                        now_mask1 = now_mask1 / (now_mask1.sum() + eps)
                        self.u_py1 = self.ema_u * self.u_py1 + (1 - self.ema_u) * now_mask1

                        def _sym_kl_local1(p, q):
                            p = torch.clamp(p, min=eps)
                            q = torch.clamp(q, min=eps)
                            return 0.5 * self.kl_div(p.log(), q) + 0.5 * self.kl_div(q.log(), p)

                        self.count_KL1[0] += _sym_kl_local1(self.py_con1, self.u_py1)
                        self.count_KL1[1] += _sym_kl_local1(self.py_uni1, self.u_py1)
                        self.count_KL1[2] += _sym_kl_local1(self.py_rev1, self.u_py1)

                # ---- optimizer step (keep your style; add scaler support if you want) ----
                self.optim.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
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
