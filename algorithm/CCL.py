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
import copy
import math

from progress.bar import Bar
from clip import clip
from model import Model

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
best_acc_co = 0
best_acc_b = 0
def sharp(a, T):
    a = a ** T
    a_sum = torch.sum(a, dim=1, keepdim=True)
    a = a / a_sum
    return a.detach()

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

def compute_py(train_loader, device):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell, _ ) in enumerate(train_loader):
        labell = labell.to(device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(device)
    return label_freq_array
    
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

        # ---- curve buffers (one value per epoch) ----
        self.curve_epoch = []
        self.curve_test_acc = []
        self.curve_entropy = []
        self.curve_n_accept = []
        self.curve_total_accept = []


        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        # self.build_black_model()
        # self.evaluator = Evaluator(cfg, self.cls_num_list)
        self.best_result = -np.inf
        self._writer = None
        self.th = cfg.th

        self.tau = 1.0
        self.py_con = compute_py(self.train_label_loader, self.device)
        self.adjustment_l1 = compute_adjustment_by_py(self.py_con, self.tau, self.device)
        self.tau_T = 0.07
        class_list = []
        for i in range(cfg.DATA.NUMBER_CLASSES):
            class_list.append(str(i))

        title = 'PEL-SSL-' + cfg.DATA.NAME
        self.logger = Logger(os.path.join(cfg.output_dir, 'logSSL.txt'), title=title)
        self.logger.set_names(['Top1 acc', 'Best Top1 acc', 'Top1 acc_b', 'Best Top1 acc_b','Top1 acc_co', 'Best Top1 acc_co','epoch'])

    def build_data_loader(self):
        cfg = self.cfg
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.DATA.NAME](
            cfg)

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
    def test_2(self):
        self.tuner.eval()
        print(f"Evaluate on the test set")
        preds = np.array([])
        targets = np.array([])

        preds_b = np.array([])
        targets_b = np.array([])

        preds_co = np.array([])
        targets_co = np.array([])
        for batch in tqdm(self.test_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            feat = self.model(image)
            output = self.tuner.head(feat)

            output_b = self.tuner.head1(feat)
            del feat

            output_co = (output + output_b) / 2.0

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())

            prob_b = F.softmax(output_b, dim=1)
            conf_b, pred_b = prob_b.max(1)
            preds_b = np.append(preds_b, pred_b.cpu().numpy())
            targets_b = np.append(targets_b, label.cpu().numpy())

            prob_co = F.softmax(output_co, dim=1)
            conf_co, pred_co = prob_co.max(1)
            preds_co = np.append(preds_co, pred_co.cpu().numpy())
            targets_co = np.append(targets_co, label.cpu().numpy())

        targets = targets.astype(int)
        preds = preds.astype(int)
        acc = sum(targets == preds) / len(targets)

        targets_b = targets_b.astype(int)
        preds_b = preds_b.astype(int)
        acc_b = sum(targets_b == preds_b) / len(targets_b)

        targets_co = targets_co.astype(int)
        preds_co = preds_co.astype(int)
        acc_co = sum(targets_co == preds_co) / len(targets_co)

        return acc, acc_b, acc_co

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

    @torch.no_grad()
    def _text_posterior(self, z: torch.Tensor, eps: float = 1e-6):
        """
        z: [B, D] normalized
        return p_text: [B, C]
        """
        z = F.normalize(z, dim=1)
        t = self.text_proto  # [C, D]
        logits = (z @ t.t()) / float(self.tau_T)
        p = F.softmax(logits, dim=1)
        return p / p.sum(dim=1, keepdim=True).clamp_min(eps)

    @torch.no_grad()
    def _moco_build(self, feat_dim: int):
        # hyper-params
        self.moco_K = getattr(self.cfg, "moco_K", 4096)
        self.moco_m = getattr(self.cfg, "moco_m", 0.999)
        self.moco_T = getattr(self.cfg, "moco_T", 0.10)

        # queue warmup
        self.moco_warmup_ratio = getattr(self.cfg, "moco_warmup_ratio", 0.25)  # 25% filled -> enable Lspl
        self.queue_filled = 0  # how many valid items are in queue (warmup gate)

        # momentum encoder
        self.model_m = copy.deepcopy(self.model)
        for p in self.model_m.parameters():
            p.requires_grad_(False)
        self.model_m.eval()

        proj = self.tuner.proj_h
        self.proj_m = None
        if proj is not None:
            self.proj_m = copy.deepcopy(proj)
            for p in self.proj_m.parameters():
                p.requires_grad_(False)
            self.proj_m.eval()

        # queues: keys + soft labels
        K, C, D = self.moco_K, self.num_classes, feat_dim
        self.queue_z = F.normalize(torch.randn(K, D, device=self.device), dim=1)
        self.queue_p = torch.full((K, C), 1.0 / C, device=self.device)
        self.queue_ptr = 0

    @torch.no_grad()
    def _moco_momentum_update(self):
        m = self.moco_m
        for p_q, p_k in zip(self.model.parameters(), self.model_m.parameters()):
            p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

        proj = self.tuner.proj_h
        if (proj is not None) and (self.proj_m is not None):
            for p_q, p_k in zip(proj.parameters(), self.proj_m.parameters()):
                p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def _moco_enqueue(self, k: torch.Tensor, p: torch.Tensor):
        """
        k: [N,D] normalized
        p: [N,C] soft labels (sum=1)
        """
        N = k.size(0)
        K = self.queue_z.size(0)
        ptr = int(self.queue_ptr)

        if N <= 0:
            return

        if N >= K:
            self.queue_z.copy_(k[-K:])
            self.queue_p.copy_(p[-K:])
            self.queue_ptr = 0
            self.queue_filled = K
            return

        end = ptr + N
        if end <= K:
            self.queue_z[ptr:end] = k
            self.queue_p[ptr:end] = p
        else:
            first = K - ptr
            self.queue_z[ptr:] = k[:first]
            self.queue_p[ptr:] = p[:first]
            rest = N - first
            self.queue_z[:rest] = k[first:]
            self.queue_p[:rest] = p[first:]
        self.queue_ptr = (ptr + N) % K
        self.queue_filled = min(K, self.queue_filled + N)

    @torch.no_grad()
    def _bank_ready(self):
        # require queue filled enough to avoid random-bank noise
        K = int(self.moco_K)
        need = int(max(1, self.moco_warmup_ratio * K))
        return getattr(self, "queue_filled", 0) >= need

    @torch.no_grad()
    def _build_bank(self):
        """
        Build current bank tensors (all detached):
        z_bank: [C + K, D]
        p_bank: [C + K, C]
        """
        z_bank = torch.cat([self.text_proto.detach(), self.queue_z.detach()], dim=0).float()
        eye = torch.eye(self.num_classes, device=self.device)
        p_bank = torch.cat([eye, self.queue_p.detach()], dim=0).float()
        return z_bank, p_bank

    @torch.no_grad()
    def _teacher_posterior_from_bank(self, q_m: torch.Tensor, p_cls: torch.Tensor,
                        z_bank: torch.Tensor, p_bank: torch.Tensor,
                        tau: float = 0.07, beta: float = 0.3, eps: float = 1e-6):
        """
        Teacher (weak view) posterior:
        - q_m in momentum space (align with queue)
        - optional mix with classifier p_cls for stability
        """
        p_nb = self._bank_posterior_from_bank(q_m, z_bank, p_bank, tau=tau, eps=eps)

        p_cls = p_cls.detach()
        p_cls = p_cls / p_cls.sum(dim=1, keepdim=True).clamp_min(eps)

        q_tilde = (1.0 - beta) * p_cls + beta * p_nb
        q_tilde = q_tilde / q_tilde.sum(dim=1, keepdim=True).clamp_min(eps)
        return q_tilde

    def _student_posterior_from_bank(self, q: torch.Tensor, z_bank: torch.Tensor, p_bank: torch.Tensor,
                        tau: float = 0.07, eps: float = 1e-6):
        """
        Student (strong view) posterior:
        - NO mix with its own classifier (closer to CCL Eq.(18))
        - grads only flow through q; bank is detached already
        """
        p_nb = self._bank_posterior_from_bank(q, z_bank, p_bank, tau=tau, eps=eps)
        return p_nb

    @torch.no_grad()
    def _bank_posterior_from_bank(self, q, z_bank, p_bank, tau=0.07, knn_k=64, eps=1e-6):
        sim = (q.float() @ z_bank.t()) / float(tau)   # [B, C+K]
        if knn_k is not None and knn_k > 0 and knn_k < sim.size(1):
            topv, topi = sim.topk(knn_k, dim=1, largest=True, sorted=False)  # [B,k]
            w = F.softmax(topv, dim=1)                                       # [B,k]
            p_nb = torch.bmm(w.unsqueeze(1), p_bank[topi].float()).squeeze(1)  # [B,C]
        else:
            w = F.softmax(sim, dim=1)
            p_nb = w @ p_bank.float()

        p_nb = p_nb / p_nb.sum(dim=1, keepdim=True).clamp_min(eps)
        return p_nb

    def train(self):
        global best_acc
        global best_acc_b
        global best_acc_co

        self.logger.set_names(['Top1 acc', 'Best Top1 acc', 'Top1 acc_b', 'Best Top1 acc_b','Top1 acc_co', 'Best Top1 acc_co','epoch'])
    
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
        self.py_u = self.py_u = (torch.ones(self.num_classes, device=self.device) / self.num_classes)
        self.time_start = time.time()
        # text anchors
        self.text_proto = F.normalize(self.text_features, dim=1)
        self._moco_build(512)
        pseudo_count = torch.zeros(self.num_classes, dtype=torch.long)  # CPU
        n_accept = 0

        # CCL hyper
        self.tau_T = 1.5
        self.tau_bank = getattr(self.cfg, "tau_bank", 1.5)   # posterior temperature
        self.beta_bank = getattr(self.cfg, "beta_bank", 0.7)  # teacher mix weight
        self.th_ccl = getattr(self.cfg, "th_ccl", self.th)    # teacher-conf threshold for Lspl
        self.w_spl = getattr(self.cfg, "w_spl", 0.3)          # weight on Lspl
        
        for self.epoch in range(self.start_epoch, self.num_epochs):
            self.tuner.train()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            self.num_batches = self.cfg.total_steps
            label_loader_iter = cycle(self.train_label_loader)
            unlabel_loader_iter = cycle(self.train_unlabel_loader)

            pyu_sum = torch.zeros(self.num_classes, device=self.device)
            pyu_cnt = 0.0 
            classwise_pace = torch.zeros(self.num_classes, device=self.device)
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
                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3 * self.cfg.DATA.MU_U + 1)

                feat = self.model(inputs)
                feat = de_interleave(feat, 3 * self.cfg.DATA.MU_U + 1)
                feat_u_w, feat_u_s, feat_u_s1 = feat[batch_size:].chunk(3)
                feat_u_w = self.tuner.proj_h_ctr(feat_u_w)

                output = self.tuner.head(feat)
                output_x = output[:batch_size]
                output_u_w, output_u_s, output_u_s1 = output[batch_size:].chunk(3)

                output1 = self.tuner.head1(feat.detach())
                output1_x = output1[:batch_size]
                output1_u_w, output1_u_s, output1_u_s1 = output1[batch_size:].chunk(3)

                del feat
                del output
                del output1

                classwise_acc = classwise_pace / classwise_pace.max().clamp(min=1e-6)
                
                mar_diff = (classwise_acc.max() - classwise_acc.min()) * self.alpha
                alpha = (((1.0 - classwise_acc) / (classwise_acc + 1.0)) * mar_diff).unsqueeze(1)
                output_x = output_x + alpha[targets_x] * self.betabase[targets_x]
                
                Lx = F.cross_entropy(output_x, targets_x, reduction='mean')
                Lx1 = F.cross_entropy(output1_x, targets_x, reduction='mean')

                pseu = torch.softmax(output_u_w.detach(), dim=-1)
                conf, targets_u = torch.max(pseu, dim=-1)
                mask = conf.ge(self.th)
                mask_twice = torch.cat([mask, mask], dim=0).to(self.device)
                output_u_s_twice = torch.cat([output_u_s, output_u_s1], dim=0).to(self.device)
                targets_u_twice = torch.cat([targets_u, targets_u], dim=0).to(self.device)

                pseu1 = torch.softmax(output1_u_w.detach(), dim=-1)
                conf1, targets1_u = torch.max(pseu1, dim=-1)
                output1_u_s_twice = torch.cat([output1_u_s, output1_u_s1], dim=0).to(self.device)
                confw = torch.cat([conf1, conf1], dim=0).to(self.device)

                mask1 = conf1.ge(self.th_min)
                mask1_twice = torch.cat([mask1, mask1], dim=0).to(self.device)

                output_u_s_twice = output_u_s_twice + alpha[targets_u_twice] * self.betabase[targets_u_twice]

                if torch.sum(mask1_twice) > 0:
                    Lu = (F.cross_entropy(output_u_s_twice, targets_u_twice,
                                          reduction='none')* confw * mask1_twice).mean()
                else:
                    Lu = 0

                onehotu = targets_u_twice.reshape(-1, 1)
                onehotu = torch.zeros_like(output1_u_s_twice).scatter(1, onehotu, 1)
                onehotu = onehotu * (1 - self.smoothing) + (1 - onehotu) * self.smoothing / (self.num_classes - 1)
                log_predu = F.log_softmax(output1_u_s_twice, dim=-1)

                log_predu = F.log_softmax(output1_u_s_twice, dim=-1)   # [B2, C]
                y = targets_u_twice                                    # [B2]
                B2, C = log_predu.shape

                # 1) remainder distribution r_y from global prior py_u
                prior = self.py_u.detach().clamp_min(1e-6)             # [C]
                r = prior.unsqueeze(0).repeat(B2, 1)                   # [B2, C]
                r.scatter_(1, y.view(-1, 1), 0.0)                      # remove true class
                r = r / r.sum(dim=1, keepdim=True).clamp_min(1e-6)     # normalize

                # 2) class-adaptive epsilon (recommended)
                #    head class -> smaller eps; tail class -> larger eps
                eps_min = getattr(self.cfg, "ls_eps_min", 0.05)
                eps_max = getattr(self.cfg, "ls_eps_max", 0.30)
                gamma   = getattr(self.cfg, "ls_gamma", 1.0)

                pu = self.py_u.detach().clamp_min(1e-6)                # [C]
                pu_norm = (pu / pu.max()).clamp(0.0, 1.0)              # [C]
                eps_class = eps_min + (eps_max - eps_min) * (1.0 - pu_norm).pow(gamma)  # [C]
                eps_i = eps_class[y]                                    # [B2]

                # 3) build soft target q
                q_one = torch.zeros_like(output1_u_s_twice).scatter(1, y.view(-1, 1), 1.0)  # [B2,C]
                q = (1.0 - eps_i.unsqueeze(1)) * q_one + eps_i.unsqueeze(1) * r             # [B2,C]

                # 4) compute Lu1 with soft targets
                per_sample = torch.sum(-q * log_predu, dim=1)          # [B2]

                # 建议：head1 的无标注监督用 mask1_twice（来自 conf1），更一致；
                # 如果你暂时不想改逻辑，也可以继续用 mask_twice
                use_mask = mask_twice  # 推荐
                # use_mask = mask_twice  # 保守（保持你当前的筛选来源）

                if torch.sum(use_mask) > 0:
                    Lu1 = per_sample[use_mask].mean()
                else:
                    Lu1 = 0

                # feat_u_w: [B, D] image features
                feat_u_w = F.normalize(feat_u_w, dim=-1)
                # text_features: [C, D] 已经 normalize
                text_sel = self.text_features[targets_u]
                semantic = (feat_u_w * text_sel).sum(dim=-1).clamp(min=0.0)  # [B]
                conf = conf * semantic

                if not hasattr(self, "conf_mu"):
                    self.conf_mu = conf.mean().detach()
                    self.conf_sigma = conf.std().detach() + 1e-6
                else:
                    self.conf_mu = 0.99 * self.conf_mu + 0.01 * conf.mean().detach()
                    self.conf_sigma = 0.99 * self.conf_sigma + 0.01 * conf.std().detach()

                soft_w = torch.exp(
                    - (conf - self.conf_mu) ** 2 / (2 * self.conf_sigma ** 2)
                ).detach()  # [B]

                for c in range(self.num_classes):
                    mask_c = (targets_u == c)
                    if mask_c.any():
                        classwise_pace[c] += soft_w[mask_c].sum()
                
                if mask.any():
                    y = targets_u[mask].detach().cpu()
                    pseudo_count += torch.bincount(y, minlength=self.num_classes)
                    n_accept += int(mask.sum().item())


                # ===================================================================
                # CCL Smoothed PL (B/C/F applied)
                # ===================================================================
                Lspl = 0.0
                if self._bank_ready():
                    # build bank once
                    z_bank, p_bank = self._build_bank()

                    # teacher cls from weak view (detach)
                    p_cls_u = pseu.detach()
                    p_cls_u = p_cls_u / p_cls_u.sum(dim=1, keepdim=True).clamp_min(1e-6)

                    # --- B: teacher query from momentum encoder (weak view) ---
                    with torch.no_grad():
                        # use current model_m (updated at last iter) as teacher space
                        ku_w = self.model_m(inputs_u_w)
                        if self.proj_m is not None:
                            ku_w = self.proj_m(ku_w)
                        ku_w = F.normalize(ku_w, dim=-1)

                        p_w = self._teacher_posterior_from_bank(
                            q_m=ku_w, p_cls=p_cls_u,
                            z_bank=z_bank, p_bank=p_bank,
                            tau=self.tau_bank, beta=self.beta_bank
                        )  # already no_grad inside

                        # --- F: teacher-based mask for Lspl ---
                        conf_w, _ = p_w.max(dim=1)
                        mask_ccl = conf_w.ge(self.th_ccl)
                    
                    # student queries from strong views (online encoder)
                    q_us  = F.normalize(self.tuner.proj_h(feat_u_s),  dim=-1)
                    q_us1 = F.normalize(self.tuner.proj_h(feat_u_s1), dim=-1)

                    # --- A: student posterior = pure bank posterior (no mix with logits) ---
                    p_s  = self._student_posterior_from_bank(q_us,  z_bank, p_bank, tau=self.tau_bank)
                    p_s1 = self._student_posterior_from_bank(q_us1, z_bank, p_bank, tau=self.tau_bank)

                    # CE / KL(p_w || p_s)
                    Lbspl  = (-p_w * torch.log(p_s.clamp_min(1e-6))).sum(dim=1)
                    Lbspl1 = (-p_w * torch.log(p_s1.clamp_min(1e-6))).sum(dim=1)

                    if mask_ccl.any():
                        Lspl = 0.5 * (Lbspl[mask_ccl].mean() + Lbspl1[mask_ccl].mean())
                    else:
                        Lspl = 0.0

                loss = (Lx + Lu * self.w_con + Lx1 + Lu1) * 0.7 + 0.3*Lspl
                if mask.any():
                    Pu_batch = pseu[mask].mean(dim=0)  # [C]
                    Pu_batch = Pu_batch / Pu_batch.sum().clamp_min(1e-4)
                    pyu_sum += Pu_batch.detach()
                    pyu_cnt += 1.0

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # ---------------- MoCo queue update (no grad) ----------------
                with torch.no_grad():
                    # momentum update AFTER online step (MoCo-style)
                    self._moco_momentum_update()

                    # run momentum encoder once for (x, u_s, u_s1)
                    m_in = torch.cat([inputs_x, inputs_u_s, inputs_u_s1], dim=0)
                    k = self.model_m(m_in)
                    if self.proj_m is not None:
                        k = self.proj_m(k)
                    k = F.normalize(k, dim=-1)

                    bx = inputs_x.size(0)
                    bu = inputs_u_s.size(0)
                    kxs, kus, kus1 = k.split([bx, bu, bu], dim=0)

                    px = F.one_hot(targets_x, num_classes=self.num_classes).float()
                    self._moco_enqueue(kxs, px)

                    if hasattr(self, "text_features") and (self.text_features is not None):
                        pt_s  = self._text_posterior(kus)   # [Bu, C]
                        pt_s1 = self._text_posterior(kus1)
                    else:
                        pt_s = pt_s1 = None

                    # still use your original mask(conf>=th) to control enqueue quality

                    if mask.any() and (pt_s is not None):
                        self._moco_enqueue(kus[mask],  pt_s[mask])
                        self._moco_enqueue(kus1[mask], pt_s1[mask])

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
            if self.epoch >= 3 and pyu_cnt > 0:
                Pu_epoch = pyu_sum / pyu_cnt
                Pu_epoch = Pu_epoch / Pu_epoch.sum().clamp_min(1e-6)
        
                m_u = getattr(self.cfg, "pyu_m", 0.9)
                self.py_u = m_u * self.py_u + (1 - m_u) * Pu_epoch
                self.py_u = self.py_u / self.py_u.sum().clamp_min(1e-6)
            last_epoch = (self.epoch + 1) == self.num_epochs
            meet_checkpoint_freq = (
                (self.epoch + 1) % self.cfg.checkpoint_freq == 0
                if self.cfg.checkpoint_freq > 0 else False
            )

            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

            # torch.cuda.empty_cache()

            acc_now, acc_b_now, acc_co_now = self.test_2()
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
            best_acc_b = max(best_acc_b, acc_b_now)
            best_acc_co = max(best_acc_co, acc_co_now)
            self.logger.append([acc_now, best_acc, acc_b_now, best_acc_b, acc_co_now, best_acc_co, self.epoch + 1])


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
            "method": "vlfd",  # 你也可以写 cfg.method
        }
        torch.save(pack, curve_path)
        print(f"[Saved] curves -> {curve_path}")

        self.logger.close()
    