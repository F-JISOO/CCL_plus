import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torchvision import transforms

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
        self.build_model()

        self.py_con = compute_py(self.train_label_loader, self.device)
        self.py_uni = torch.ones(self.num_classes) / self.num_classes
        self.py_rev = torch.flip(self.py_con, dims=[0])

        self.py_uni = self.py_uni.to(self.device)


        self.count_KL = torch.zeros(3).to(self.device)

        self.tau1 = 2
        self.T = 1
        self.tau12 = 2
        self.tau2 = 2
        self.est_epoch = 5
        self.eval_step = 500
        self.threshold = 0.7

        self.adjustment_l1 = compute_adjustment_by_py(self.py_con, self.tau1, self.device)
        self.adjustment_l12 = compute_adjustment_by_py(self.py_con, self.tau12, self.device)
        self.adjustment_l2 = compute_adjustment_by_py(self.py_con, self.tau2, self.device)

        self.taumin = 0
        self.taumax = self.tau1
        
        self.u_py = torch.ones(self.num_classes) / self.num_classes
        self.ema_u = 0.99
        self.u_py = self.u_py.to(self.device)
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

    def build_data_loader(self):
        cfg = self.cfg
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.DATA.NAME](
            cfg)

        self.num_classes = cfg.DATA.NUMBER_CLASSES
        self.classnames = labeled_dataset.classes

        # self.sampled_cls_num_list = self.cls_num_list
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
        KL_div = nn.KLDivLoss(reduction='sum')
        for self.epoch in range(self.start_epoch, self.num_epochs):
            self.tuner.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            self.num_batches = self.cfg.total_steps
            label_loader_iter = cycle(self.train_label_loader)
            unlabel_loader_iter = cycle(self.train_unlabel_loader)

            if self.epoch > self.est_epoch:
                self.count_KL = self.count_KL / self.eval_step
            KL_softmax = (torch.exp(self.count_KL[0])) / (torch.exp(self.count_KL[0])+torch.exp(self.count_KL[1])+torch.exp(self.count_KL[2]))
            tau = self.taumin + (self.taumax - self.taumin) * KL_softmax
            if math.isnan(tau) == False:
                self.adjustment_l1 = compute_adjustment_by_py(self.py_con, tau, self.device)

            self.count_KL = torch.zeros(3).to(self.device)

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
                u_real = u_real.to(self.device)
                mask_l = (u_real != -2)
                mask_l = mask_l.to(self.device)

                feat = self.model(inputs)
                feat = de_interleave(feat, 3 * self.cfg.DATA.MU_U + 1)

                output = self.tuner.head(feat)
                output_x = output[:batch_size]
                output_u_w, output_u_s, output_u_s1 = output[batch_size:].chunk(3)

                output1 = self.tuner.head1(feat.detach())
                output1_x = output1[:batch_size]
                output1_u_w, output1_u_s, output1_u_s1 = output1[batch_size:].chunk(3)

                del feat
                del output
                del output1

                Lx = F.cross_entropy(output_x, targets_x, reduction='mean')
                Lx_b = F.cross_entropy(output1_x + self.adjustment_l2, targets_x, reduction='mean')

                pseudo_label = torch.softmax((output_u_w.detach() - self.adjustment_l1) / self.T, dim=-1)
                pseudo_label_h2 = torch.softmax((output_u_w.detach() - self.adjustment_l12) / self.T, dim=-1)
                pseudo_label_b = torch.softmax(output1_u_w.detach() / self.T, dim=-1)
                pseudo_label_t = torch.softmax(output_u_w.detach() / self.T, dim=-1)

                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)
                max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)
                max_probs_t, targets_u_t = torch.max(pseudo_label_t, dim=-1)

                mask = max_probs.ge(self.threshold)
                mask_h2 = max_probs_h2.ge(self.threshold)
                mask_b = max_probs_b.ge(self.threshold)
                mask_t = max_probs_t.ge(self.threshold)

                mask_ss_b_h2 = mask_b + mask_h2
                mask_ss_t = mask + mask_t

                mask = mask.float()
                mask_b = mask_b.float()

                mask_ss_b_h2 = mask_ss_b_h2.float()
                mask_ss_t = mask_ss_t.float()

                mask_twice_ss_b_h2 = torch.cat([mask_ss_b_h2, mask_ss_b_h2], dim=0).to(self.device)
                mask_twice_ss_t = torch.cat([mask_ss_t, mask_ss_t], dim=0).to(self.device)

                logits_u_s_twice = torch.cat([output_u_s, output_u_s1], dim=0).to(self.device)
                targets_u_twice = torch.cat([targets_u, targets_u], dim=0).to(self.device)
                targets_u_h2_twice = torch.cat([targets_u_h2, targets_u_h2], dim=0).to(self.device)

                logits_u_s_b_twice = torch.cat([output1_u_s, output1_u_s1], dim=0).to(self.device)

                now_mask = torch.zeros(self.num_classes)
                now_mask = now_mask.to(self.device)
                u_real[u_real == -2] = 0

                if self.epoch > self.est_epoch:
                    now_mask[targets_u_b] += mask_l*mask_b

                    if now_mask.sum() > 0:
                        now_mask = now_mask / now_mask.sum()
                        self.u_py = self.ema_u * self.u_py + (1-self.ema_u) * now_mask
                        KL_con = 0.5 * KL_div(self.py_con.log(), self.u_py) + 0.5 * KL_div(self.u_py.log(), self.py_con)
                        KL_uni = 0.5 * KL_div(self.py_uni.log(), self.u_py) + 0.5 * KL_div(self.u_py.log(), self.py_uni)
                        KL_rev = 0.5 * KL_div(self.py_rev.log(), self.u_py) + 0.5 * KL_div(self.u_py.log(), self.py_rev)
                        self.count_KL[0] = self.count_KL[0] + KL_con
                        self.count_KL[1] = self.count_KL[1] + KL_uni
                        self.count_KL[2] = self.count_KL[2] + KL_rev

                Lu = (F.cross_entropy(logits_u_s_twice, targets_u_twice,
                                    reduction='none') * mask_twice_ss_t).mean()
                Lu_b = (F.cross_entropy(logits_u_s_b_twice, targets_u_h2_twice,
                                        reduction='none') * mask_twice_ss_b_h2).mean()
                loss = Lx + Lu + Lx_b + Lu_b

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
