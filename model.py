import math
from operator import mul
from functools import reduce
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

import copy


class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)
        
    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, adapter_scalar=0.1, scalar_learnable=False, adapter_type='parallel', dtype=None):
        super().__init__()
        self.adapter_type = adapter_type

        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

        # self.scale = float(adapter_scalar)
        if scalar_learnable is True:
            # self.scale = nn.Parameter(torch.ones(1))
            self.scale = nn.Parameter(torch.ones([]) * adapter_scalar)
        else:
            self.scale = float(adapter_scalar)

    def forward(self, x):
        out = self.ln(x)
        out = self.down_proj(out)
        out = self.relu(out)
        out = self.up_proj(out)

        out = out * self.scale

        if self.adapter_type == 'sequential':
            return out
        elif self.adapter_type == 'parallel':
            return out + x


class AdaptFormer(nn.Module):
    def __init__(self,
                 d_model=768,
                 bottleneck=64,
                 dropout=0.1,
                 # init_option="bert",
                 init_option="lora",
                 adapter_scalar=0.1,
                 scalar_learnable=False,
                 adapter_layernorm_option="in", dtype=None):
        super().__init__()
        # self.n_embd = config.d_model if d_model is None else d_model
        # self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd, dtype=dtype)

        if scalar_learnable is True:
            # self.scale = nn.Parameter(torch.ones(1))
            self.scale = nn.Parameter(torch.ones([]) * adapter_scalar)
        else:
            self.scale = float(adapter_scalar)

        # self.scale = nn.Parameter(torch.ones(1)).to(dtype)

        self.down_proj = nn.Linear(self.n_embd, self.down_size, dtype=dtype)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd, dtype=dtype)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim, dtype=dtype))
        self.scaling = 1.0 / bottle_dim
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    def forward(self, x):
        return x * self.scale + self.shift


class multi_SSF(nn.Module):
    def __init__(self, n_cls, in_dim, init_mean=1.0, init_std=0.02, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_cls, in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(n_cls, in_dim, dtype=dtype))
        # nn.init.normal_(self.scale, mean=1.0, std=0.02)
        # nn.init.normal_(self.shift, std=0.02)
        nn.init.normal_(self.scale, mean=init_mean, std=init_std)
        nn.init.normal_(self.shift, std=init_std)

    def forward(self, x, norm=False):
        if norm is True:
            x = x / x.norm(dim=-1, keepdim=True)
        return x * self.scale + self.shift


class ViT_Head(nn.Module):
    # def __init__(self, text_features, visual_proj, logit_scale, rand_init=False):
    def __init__(self, num_classes, visual_proj, emb_dim, text_features=None):
        super().__init__()
        # in_dim = visual_proj.shape[0]
        # n_cls = text_features.shape[0]
        n_cls = num_classes

        self.weight = nn.Parameter(torch.empty(n_cls, emb_dim))

        if text_features is not None:
            self.weight.data = text_features.data @ visual_proj.data.t()
        else:
            self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        # self.logit_scale = nn.Parameter(logit_scale.exp().clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        return self.logit_scale * (x @ weight.t())


class ViT_Head_v0(nn.Module):
    def __init__(self, text_features, visual_proj, logit_scale):
        super().__init__()
        self.visual_proj = nn.Parameter(visual_proj.clone())
        self.weight = nn.Parameter(text_features.clone())
        # self.logit_scale = nn.Parameter(logit_scale.exp().clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

    def forward(self, x):
        x = x @ self.visual_proj
        x = x / x.norm(dim=-1, keepdim=True)
        weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        return self.logit_scale * (x @ weight.t())


class MetaExpertFuse(nn.Module):
    """
    Fully copied from Meta-Expert official MetaExpertNet.forward() fusion logic:
    - MFF: lin1~lin8 residual feature fusion (two-stage)
    - DEA: predict() over [feat444, lp1, lp2, lp3]
    - gating: p1/p2/p3 reweight then softmax again
    - outputs: fuse_w_logit, w1/w2/w3, fuse_logit_l, fuse_logit_HMT_c_1/2/3
    """
    def __init__(self, channels, num_classes, cut1, cut2):
        """
        channels: list/tuple of 4 dims [c1,c2,c3,c4] for feat1..feat4
                  (CLIP ViT-B: all 768 is OK -> [768,768,768,768])
        """
        super().__init__()
        self.channels = list(channels)
        self.cut1 = int(cut1)
        self.cut2 = int(cut2)
        self.num_classes = int(num_classes)

        # --- MFF: stage-1 ---
        self.lin1 = nn.Sequential(nn.Linear(self.channels[0], self.channels[1]), nn.SiLU())
        self.lin2 = nn.Sequential(nn.Linear(self.channels[1], self.channels[2]), nn.SiLU())
        self.lin3 = nn.Sequential(nn.Linear(self.channels[2], self.channels[3]), nn.SiLU())
        self.lin4 = nn.Sequential(nn.Linear(self.channels[3], 2 * self.channels[3]), nn.SiLU())

        # --- MFF: stage-2 ---
        self.lin5 = nn.Sequential(nn.Linear(self.channels[1], self.channels[2]), nn.SiLU())
        self.lin6 = nn.Sequential(nn.Linear(self.channels[2], self.channels[3]), nn.SiLU())
        self.lin7 = nn.Sequential(nn.Linear(self.channels[3], 2 * self.channels[3]), nn.SiLU())
        self.lin8 = nn.Sequential(nn.Linear(2 * self.channels[3], 2 * self.channels[3]), nn.SiLU())

        # --- DEA predictor (exact dims & SiLU tail as official) ---
        self.predict = nn.Sequential(
            nn.Linear(2 * self.channels[3] + 3 * self.num_classes, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 3), nn.SiLU()
        )
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _lnn(z: torch.Tensor) -> torch.Tensor:
        # official: subtract row-wise min
        return z - torch.min(z, dim=1, keepdim=True).values

    def forward(self, outputs: dict, feat_for_fuse: dict, p_hat_lb: torch.Tensor, tau1, tau2, tau3):
        # ---- unpack feats ----
        feat1 = feat_for_fuse["feat1"]
        feat2 = feat_for_fuse["feat2"]
        feat3 = feat_for_fuse["feat3"]
        feat4 = feat_for_fuse["feat4"]

        # ---- 3 heads logits ----
        c_logit_1 = outputs["logits"]
        c_logit_2 = outputs["aux_logits1"]
        c_logit_3 = outputs["aux_logits2"]

        cp_logit_1 = self.softmax(c_logit_1)
        cp_logit_2 = self.softmax(c_logit_2)
        cp_logit_3 = self.softmax(c_logit_3)

        # ---- expert logits (H/M/T for 3 heads) ----
        c_logit_x_H_1 = outputs["logitsH"]
        c_logit_x_M_1 = outputs["logitsM"]
        c_logit_x_T_1 = outputs["logitsT"]

        c_logit_x_H_2 = outputs["aux_logitsH1"]
        c_logit_x_M_2 = outputs["aux_logitsM1"]
        c_logit_x_T_2 = outputs["aux_logitsT1"]

        c_logit_x_H_3 = outputs["aux_logitsH2"]
        c_logit_x_M_3 = outputs["aux_logitsM2"]
        c_logit_x_T_3 = outputs["aux_logitsT2"]

        # ---- LNN for expert logits (exact) ----
        lnn_c_logit_x_H_1 = self._lnn(c_logit_x_H_1)
        lnn_c_logit_x_M_1 = self._lnn(c_logit_x_M_1)
        lnn_c_logit_x_T_1 = self._lnn(c_logit_x_T_1)

        lnn_c_logit_x_H_2 = self._lnn(c_logit_x_H_2)
        lnn_c_logit_x_M_2 = self._lnn(c_logit_x_M_2)
        lnn_c_logit_x_T_2 = self._lnn(c_logit_x_T_2)

        lnn_c_logit_x_H_3 = self._lnn(c_logit_x_H_3)
        lnn_c_logit_x_M_3 = self._lnn(c_logit_x_M_3)
        lnn_c_logit_x_T_3 = self._lnn(c_logit_x_T_3)

        # ---- logit-adjusted labeled logits (exact) ----
        hat = p_hat_lb.to(device=c_logit_1.device, dtype=c_logit_1.dtype).clamp_min(1e-12)
        l_logit_1 = c_logit_1 + float(tau1) * torch.log(hat)
        l_logit_2 = c_logit_2 + float(tau2) * torch.log(hat)
        l_logit_3 = c_logit_3 + float(tau3) * torch.log(hat)

        lp_logit_1 = self.softmax(l_logit_1)
        lp_logit_2 = self.softmax(l_logit_2)
        lp_logit_3 = self.softmax(l_logit_3)

        lnn_l_logit_1 = self._lnn(l_logit_1)
        lnn_l_logit_2 = self._lnn(l_logit_2)
        lnn_l_logit_3 = self._lnn(l_logit_3)

        # ===================== MFF (exact) =====================
        feat11 = self.lin1(feat1)
        feat22 = self.lin2(feat2 + feat11)
        feat33 = self.lin3(feat3 + feat22)
        feat44 = self.lin4(feat4 + feat33)

        feat111 = self.lin5(feat11)
        feat222 = self.lin6(feat22 + feat111)
        feat333 = self.lin7(feat33 + feat222)
        feat444 = self.lin8(feat44 + feat333)

        # ===================== DEA (exact) =====================
        fuse_out = torch.cat([feat444, lp_logit_1, lp_logit_2, lp_logit_3], dim=1)
        out_for_attention = self.predict(fuse_out)  # [N,3]

        w1, w2, w3 = self.softmax(out_for_attention).chunk(3, dim=1)

        p1 = torch.max(cp_logit_1[:, :self.cut1], dim=1, keepdim=True).values
        p2 = torch.max(cp_logit_2[:, self.cut1:self.cut2], dim=1, keepdim=True).values
        p3 = torch.max(cp_logit_3[:, self.cut2:], dim=1, keepdim=True).values

        new_w1, new_w2, new_w3 = self.softmax(torch.cat([p1 * w1, p2 * w2, p3 * w3], dim=1)).chunk(3, dim=1)

        # ---- write back to outputs (exact keys) ----
        outputs["fuse_w_logit"] = out_for_attention
        outputs["w1"], outputs["w2"], outputs["w3"] = new_w1, new_w2, new_w3

        outputs["fuse_logit_l"] = outputs["w1"] * lnn_l_logit_1 + outputs["w2"] * lnn_l_logit_2 + outputs["w3"] * lnn_l_logit_3

        outputs["fuse_logit_HMT_c_1"] = outputs["w1"] * lnn_c_logit_x_H_1 + outputs["w2"] * lnn_c_logit_x_M_1 + outputs["w3"] * lnn_c_logit_x_T_1
        outputs["fuse_logit_HMT_c_2"] = outputs["w1"] * lnn_c_logit_x_H_2 + outputs["w2"] * lnn_c_logit_x_M_2 + outputs["w3"] * lnn_c_logit_x_T_2
        outputs["fuse_logit_HMT_c_3"] = outputs["w1"] * lnn_c_logit_x_H_3 + outputs["w2"] * lnn_c_logit_x_M_3 + outputs["w3"] * lnn_c_logit_x_T_3

        return outputs
    

class ViT_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        n_layers = clip_model.visual.transformer.layers
        emb_dim = clip_model.visual.transformer.width
        seq_len = clip_model.visual.positional_embedding.shape[0]
        patch_size = clip_model.visual.conv1.kernel_size
        dtype = clip_model.dtype

        use_finetune = cfg.finetune
        use_bias_tuning = cfg.bias_tuning
        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_vpt_last = cfg.vpt_last

        use_adapter = cfg.adapter
        use_lora = cfg.lora
        use_ssf = cfg.ssf
        vpt_len = cfg.vpt_len
        adapter_dim = cfg.adapter_dim
        lora_dim = cfg.lora_dim
        partial = cfg.partial
        block_num = cfg.block_num

        use_adaptformer = cfg.adaptformer
        ffn_num = cfg.ffn_num

        if partial is None:
            partial = n_layers
        else:
            partial = int(partial)

        block_list = []
        if block_num is not None:
            block_num = int(block_num)
            init_block = n_layers - 1
            for i in range(block_num):
                block_list.append(init_block - i)

        blocks = clip_model.visual.transformer.resblocks

        if use_finetune:
            if block_num is None:
                finetune_list = nn.ParameterList([
                    param for name, param in clip_model.visual.named_parameters()
                ])
            else:
                finetune_list = nn.ParameterList([])
                for name, param in clip_model.visual.named_parameters():
                    for num in block_list:
                        if 'resblocks.' + str(num) in name:
                            finetune_list.append(param)
                            continue
        else:
            finetune_list = None

        if use_bias_tuning:
            bias_list = nn.ParameterList([
                param for name, param in clip_model.visual.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_list = None

        assert int(use_vpt_shallow) + int(use_vpt_deep) < 2
        if use_vpt_shallow:
            vpt_list = nn.ModuleList([
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype),
                *[None] * (n_layers - 1)
            ])
        elif use_vpt_deep:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype) for _ in range(partial)]
            ])
        elif use_vpt_last:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - 1),
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype)
            ])
        else:
            vpt_list = [None] * n_layers

        # if use_adapter:
        #     adapter_list = nn.ModuleList([
        #         *[None] * (n_layers - partial),
        #         *[Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(partial)]
        #     ])
        # else:
        #     adapter_list = [None] * n_layers

        if use_adapter:
            adapter_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, adapter_scalar=cfg.adapter_scalar,
                          scalar_learnable=cfg.scalar_learnable, dtype=dtype) for _ in range(partial)]
            ])
        else:
            adapter_list = [None] * n_layers

        if use_lora:
            lora_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[nn.ModuleDict({
                    "q": LoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype),
                    "v": LoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            lora_list = [None] * n_layers

        if use_ssf:
            _block_0 = clip_model.visual.transformer.resblocks[0]
            ssf_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[nn.ModuleDict({
                    "attn_in": SSF(_block_0.attn.in_proj_bias.shape[0], dtype=dtype),
                    "attn_out": SSF(_block_0.attn.out_proj.bias.shape[0], dtype=dtype),
                    "mlp_in": SSF(_block_0.mlp[0].bias.shape[0], dtype=dtype),
                    "mlp_out": SSF(_block_0.mlp[2].bias.shape[0], dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            ssf_list = [None] * n_layers

        if use_adaptformer:
            adaptformer_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[AdaptFormer(d_model=emb_dim, bottleneck=ffn_num, adapter_scalar=cfg.adapter_scalar, scalar_learnable=cfg.scalar_learnable,
                              adapter_layernorm_option=cfg.ln_opt, dtype=dtype) for _ in
                  range(partial)]
            ])
        else:
            adaptformer_list = [None] * n_layers

        visual_proj = clip_model.visual.proj.data
        logit_scale = clip_model.logit_scale.data
        # coarse(superclass) head: K = ceil(C/4)
        K = int(getattr(cfg, "NUM_SUPER", math.ceil(cfg.DATA.NUMBER_CLASSES / 4.0)))

        # 注意：ViT_Head 里 text_features 初始化要求 shape=[n_cls, D]，
        # 你的 text_features 是 [C, D]，不能拿来 init [K, *] 的 heads/heads1
        heads  = ViT_Head(K, visual_proj, 768, text_features=None).to(clip_model.dtype)
        heads1 = ViT_Head(K, visual_proj, 768, text_features=None).to(clip_model.dtype)
        # head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)

        if cfg.rand_init:
            head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_H1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_M1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_T1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
        else:
            head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_H1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_M1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_T1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)

        if cfg.rand_init1:
            head1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_H2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_M2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_T2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)

        else:
            head1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_H2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_M2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_T2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)

        if cfg.rand_init1:
            head2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_H3 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_M3 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
            head_T3 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
        else:
            head2 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_H3 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_M3 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)
            head_T3 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)

        # To be optimized
        self.finetune_list = finetune_list
        self.bias_list = bias_list
        self.vpt_list = vpt_list
        self.adapter_list = adapter_list
        self.lora_list = lora_list
        self.ssf_list = ssf_list
        self.head = head
        self.head1 = head1
        self.head2 = head2
        self.head_H1 = head_H1
        self.head_H2 = head_H2
        self.head_H3 = head_H3
        self.head_M1 = head_M1
        self.head_M2 = head_M2
        self.head_M3 = head_M3
        self.head_T1 = head_T1
        self.head_T2 = head_T2
        self.head_T3 = head_T3
        
        self.heads = heads
        self.heads1 = heads1
        self.meta_fuse = MetaExpertFuse(
                channels=[768, 768, 768, 768],   # 你现在 feat1..feat4 都是 CLS 768
                num_classes=cfg.DATA.NUMBER_CLASSES,
                cut1=cfg.cut1,
                cut2=cfg.cut2,
            ).half()
        self.proj_h = nn.Sequential(nn.Linear(768, 768), nn.BatchNorm1d(768),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(768, 512))
        self.proj_h = self.proj_h.half()  #  FP16

        self.proj_h_ctr = nn.Sequential(nn.Linear(768, 768), nn.BatchNorm1d(768),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(768, 512))
        
        self.proj_h_ctr = self.proj_h_ctr.half()  #  FP16

        self.proj_c = nn.Sequential(nn.Linear(768, 768), nn.BatchNorm1d(768),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(768, 512))
        self.proj_c = self.proj_c.half()  #  FP16

        self.adaptformer_list = adaptformer_list
        self.ffn_opt = cfg.ffn_opt

        self.proj = copy.deepcopy(clip_model.visual.proj)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

        self.alpha_mat = nn.Parameter(torch.ones((cfg.DATA.NUMBER_CLASSES, cfg.DATA.NUMBER_CLASSES), dtype=dtype) * cfg.alpha)
        self.text_emb = nn.Parameter(text_features.clone())

        self.alpha_cls = nn.Parameter(torch.ones((cfg.DATA.NUMBER_CLASSES, ), dtype=dtype) * cfg.alpha)
        self.alpha = nn.Parameter(torch.ones([]) * cfg.alpha)
    
    @torch.no_grad()
    def _assert_keys(self, d):
        need = ["logits","aux_logits1","aux_logits2",
                "logitsH","logitsM","logitsT",
                "aux_logitsH1","aux_logitsM1","aux_logitsT1",
                "aux_logitsH2","aux_logitsM2","aux_logitsT2"]
        for k in need:
            assert k in d, f"missing key: {k}"

    def forward_metaexpert(self, feat, feat_for_fuse, p_hat_lb, tau1, tau2, tau3):
        out = {
            "logits": self.head(feat),
            "aux_logits1": self.head1(feat),
            "aux_logits2": self.head2(feat),

            "logitsH": self.head_H1(feat),
            "logitsM": self.head_M1(feat),
            "logitsT": self.head_T1(feat),

            "aux_logitsH1": self.head_H2(feat),
            "aux_logitsM1": self.head_M2(feat),
            "aux_logitsT1": self.head_T2(feat),

            "aux_logitsH2": self.head_H3(feat),
            "aux_logitsM2": self.head_M3(feat),
            "aux_logitsT2": self.head_T3(feat),
        }
        out = self.meta_fuse(out, feat_for_fuse, p_hat_lb=p_hat_lb, tau1=tau1, tau2=tau2, tau3=tau3)
        return out

class CLIP_ViT(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        self.dtype = torch.float16

    # def forward(self, x, tuner=None):
    #     x = x.to(self.dtype)
    #     x = self.conv1(x)  # shape = [*, width, grid, grid]
    #     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    #     x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    #     x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    #     x = x + self.positional_embedding.to(device=x.device, dtype=x.dtype)
    #     x = self.ln_pre(x)

    #     _bsz = x.shape[0]
    #     _seq_len = x.shape[1]
    #     _emb_dim = x.shape[2]

    #     n_layers = self.transformer.layers

    #     for i in range(n_layers):
    #         block = self.transformer.resblocks[i]

    #         if tuner is not None:
    #             vpt = tuner.vpt_list[i]
    #             adapter = tuner.adapter_list[i]
    #             lora = tuner.lora_list[i]
    #             ssf = tuner.ssf_list[i]
    #             adaptformer = tuner.adaptformer_list[i]
    #         else:
    #             vpt = adapter = lora = ssf = adaptformer = None

    #         if vpt is not None:
    #             x = vpt(x)

    #         _seq_len_after_vpt = x.shape[1]

    #         x = x.permute(1, 0, 2)  # NLD -> LND

    #         _attn = block.attn
    #         _ln_1 = block.ln_1
    #         _mlp = block.mlp
    #         _ln_2 = block.ln_2

    #         _attn_in_proj_weight = _attn.in_proj_weight
    #         _attn_in_proj_bias = _attn.in_proj_bias
    #         _attn_out_proj_weight = _attn.out_proj.weight
    #         _attn_out_proj_bias = _attn.out_proj.bias
    #         _mlp_in_proj = _mlp[0]
    #         _mlp_gelu = _mlp[1]
    #         _mlp_out_proj = _mlp[2]

    #         _num_heads = _attn.num_heads
    #         _head_dim = _emb_dim // _num_heads

    #         ###############################
    #         ## Multi-Head Self-Attention ##
    #         ###############################
    #         residual = x  # deep copy

    #         x = _ln_1(x)

    #         qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
    #         if ssf is not None:
    #             qkv = ssf["attn_in"](qkv)
    #         q, k, v = qkv.chunk(3, dim=-1)

    #         if lora is not None:
    #             q = q + lora["q"](x)
    #             v = v + lora["v"](x)

    #         q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
    #         k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
    #         v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
    #         q = q / math.sqrt(_head_dim)
    #         attn = torch.bmm(q, k.transpose(-2, -1))
    #         attn = F.softmax(attn, dim=-1)
    #         x = torch.bmm(attn, v)
    #         x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            
    #         x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
    #         if ssf is not None:
    #             x = ssf["attn_out"](x)
    #         x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

    #         x = residual + x

    #         if adaptformer is not None:
    #             adapt_x = adaptformer(x, add_residual=False)

    #         ##########################
    #         ## Feed-Forward Network ##
    #         ##########################
    #         residual = x  # deep copy

    #         x = _ln_2(x)

    #         x = _mlp_in_proj(x)
    #         if ssf is not None:
    #             x = ssf["mlp_in"](x)
    #         x = _mlp_gelu(x)
    #         x = _mlp_out_proj(x)
    #         if ssf is not None:
    #             x = ssf["mlp_out"](x)
            
    #         if adapter is not None:
    #             x = adapter(x)

    #         if adaptformer is not None:
    #             if tuner.ffn_opt == "parallel":
    #                 x = x + adapt_x
    #             elif tuner.ffn_opt == "sequential":
    #                 x = adaptformer(x)
            
    #         x = residual + x
            
    #         x = x.permute(1, 0, 2)  # LND -> NLD

    #     x = self.ln_post(x[:, 0, :])
    #     return x
    def forward(self, x, tuner=None, return_layers=None):
        """
        return_layers: list[int]，例如 [2,5,8,11]，表示在这些 resblock 结束后取 CLS
        返回:
        - 如果 return_layers is None: 只返回 final_feat: [N, 768]
        - 否则: (final_feat, feats_dict)，feats_dict[layer_id] = [N, 768]
        """
        x = x.to(self.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )
        x = x + self.positional_embedding.to(device=x.device, dtype=x.dtype)
        x = self.ln_pre(x)

        _bsz, _emb_dim = x.shape[0], x.shape[2]
        n_layers = self.transformer.layers

        # 用 dict 存：key=layer_id, value=cls_feat
        feats_dict = {} if return_layers is not None else None
        return_set = set(return_layers) if return_layers is not None else None

        for i in range(n_layers):
            block = self.transformer.resblocks[i]

            if tuner is not None:
                vpt = tuner.vpt_list[i]
                adapter = tuner.adapter_list[i]
                lora = tuner.lora_list[i]
                ssf = tuner.ssf_list[i]
                adaptformer = tuner.adaptformer_list[i]
            else:
                vpt = adapter = lora = ssf = adaptformer = None

            if vpt is not None:
                x = vpt(x)

            _seq_len_after_vpt = x.shape[1]
            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj = _mlp[0]
            _mlp_gelu = _mlp[1]
            _mlp_out_proj = _mlp[2]

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            # ---- MHSA ----
            residual = x
            x = _ln_1(x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            if ssf is not None:
                qkv = ssf["attn_in"](qkv)
            q, k, v = qkv.chunk(3, dim=-1)

            if lora is not None:
                q = q + lora["q"](x)
                v = v + lora["v"](x)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)

            q = q / math.sqrt(_head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            x = torch.bmm(attn, v)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)

            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            if ssf is not None:
                x = ssf["attn_out"](x)
            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

            x = residual + x

            if adaptformer is not None:
                adapt_x = adaptformer(x, add_residual=False)

            # ---- FFN ----
            residual = x
            x = _ln_2(x)

            x = _mlp_in_proj(x)
            if ssf is not None:
                x = ssf["mlp_in"](x)
            x = _mlp_gelu(x)
            x = _mlp_out_proj(x)
            if ssf is not None:
                x = ssf["mlp_out"](x)

            if adapter is not None:
                x = adapter(x)

            if adaptformer is not None:
                if tuner.ffn_opt == "parallel":
                    x = x + adapt_x
                elif tuner.ffn_opt == "sequential":
                    x = adaptformer(x)

            x = residual + x
            x = x.permute(1, 0, 2)  # LND -> NLD

            # ====== 关键：在 block i 结束后取 CLS ======
            if return_set is not None and i in return_set:
                feats_dict[i] = self.ln_post(x[:, 0, :])  # [N,768]

        final_feat = self.ln_post(x[:, 0, :])  # [N,768]
        if feats_dict is None:
            return final_feat
        return final_feat, feats_dict

class RN_Head(nn.Module):
    def __init__(self, text_features, logit_scale):
        super().__init__()
        n_cls = text_features.shape[0]
        self.weight = nn.Parameter(text_features.clone())
        # self.logit_scale = nn.Parameter(logit_scale.exp().clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))
        # self.logit_scale = torch.ones([]) * (1 / 0.05)

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        return self.logit_scale * (x @ weight.t())


class RN_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        out_dim = clip_model.visual.output_dim
        dtype = clip_model.dtype

        use_finetune = cfg.finetune
        use_bias_tuning = cfg.bias_tuning
        use_bn_tuning = cfg.bn_tuning
        use_ssf = cfg.ssf

        blocks = nn.Sequential(*[
            *clip_model.visual.layer1,
            *clip_model.visual.layer2,
            *clip_model.visual.layer3,
            *clip_model.visual.layer4,
        ])

        if use_finetune:
            finetune_list = nn.ParameterList([
                param for name, param in clip_model.visual.named_parameters()
            ])
        else:
            finetune_list = None

        if use_bias_tuning:
            bias_list = nn.ParameterList([
                param for name, param in clip_model.visual.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_list = None

        if use_bn_tuning:
            bn_list = nn.ModuleList([
                clip_model.visual.bn1,
                clip_model.visual.bn2,
                clip_model.visual.bn3,
                *[nn.ModuleList([
                    block.bn1,
                    block.bn2,
                    block.bn3,
                ]) for block in blocks],
            ])
        else:
            bn_list = None

        if use_ssf:
            ssf_list = nn.ModuleList([
                SSF(out_dim, dtype=dtype),
            ])
        else:
            ssf_list = None

        logit_scale = clip_model.logit_scale.data
        head = RN_Head(text_features, logit_scale).to(clip_model.dtype)

        # To be optimized
        self.finetune_list = finetune_list
        self.bias_list = bias_list
        self.bn_list = bn_list
        self.ssf_list = ssf_list
        self.head = head

class CLIP_RN(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.bn1 = clip_model.visual.bn1
        self.conv2 = clip_model.visual.conv2
        self.bn2 = clip_model.visual.bn2
        self.conv3 = clip_model.visual.conv3
        self.bn3 = clip_model.visual.bn3
        self.avgpool = clip_model.visual.avgpool
        self.relu = clip_model.visual.relu
        self.layer1 = clip_model.visual.layer1
        self.layer2 = clip_model.visual.layer2
        self.layer3 = clip_model.visual.layer3
        self.layer4 = clip_model.visual.layer4
        self.attnpool = clip_model.visual.attnpool
        self.dtype = clip_model.dtype
    
    def forward(self, x, tuner=None):
        
        x = x.to(self.dtype)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        ssf_list = tuner.ssf_list
        if ssf_list is not None:
            x = ssf_list[0](x)
        
        return x


class Model(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        if cfg.backbone.startswith("ViT"):
            self.image_encoder = CLIP_ViT(clip_model)
            self.tuner = ViT_Tuner(cfg, clip_model, text_features)
        else:
            self.image_encoder = CLIP_RN(clip_model)
            self.tuner = RN_Tuner(cfg, clip_model)

    # def forward(self, image):
    #     feat = self.image_encoder(image, self.tuner)
    #     return feat
    def forward(self, image, return_layers=None):
        if return_layers is None:
            return self.image_encoder(image, self.tuner)
        # 只对 ViT 分支透传 return_layers
        return self.image_encoder(image, self.tuner, return_layers=return_layers)
    
class Model_linear(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        if cfg.backbone.startswith("ViT"):
            # self.image_encoder = clip_model.visual
            self.image_encoder = CLIP_ViT(clip_model)
            self.tuner = Linear_VIT_Tuner(cfg, clip_model, text_features)
        else:
            self.image_encoder = clip_model.visual
            self.tuner = Linear_RN_Tuner(cfg, clip_model, text_features)

    def forward(self, image):
        image = image.to(torch.float16)
        feat = self.image_encoder(image)
        return feat
    

class Linear_VIT_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        emb_dim = clip_model.visual.transformer.width
        dtype = torch.float16
        ffn_num = cfg.ffn_num
        lora_dim = cfg.lora_dim
        visual_proj = clip_model.visual.proj.data

        dtype = torch.float16
        if cfg.rand_init:
            head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(dtype)
        else:
            head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(dtype)

        if cfg.rand_init1:
            head1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(dtype)
        else:
            head1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(dtype)

        # To be optimized
        self.head = head
        self.head1 = head1

        
        # self.learn = DoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype)
        self.learn = AdaptFormer(d_model=emb_dim, bottleneck=ffn_num, adapter_scalar=cfg.adapter_scalar, scalar_learnable=cfg.scalar_learnable,
                              adapter_layernorm_option=cfg.ln_opt, dtype=dtype)
        # self.learn = AdaptFormer_wo_relu_layer(d_model=emb_dim, bottleneck=ffn_num, adapter_scalar=cfg.adapter_scalar, scalar_learnable=cfg.scalar_learnable,
        #                       adapter_layernorm_option=cfg.ln_opt, dtype=dtype)

class Linear_RN_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        emb_dim = clip_model.visual.output_dim
        dtype = torch.float16
        ffn_num = cfg.ffn_num
        logit_scale = clip_model.logit_scale.data

        dtype = torch.float16
        if cfg.rand_init:
            head = RN_Head(text_features, logit_scale).to(dtype)
        else:
            head = RN_Head(text_features, logit_scale).to(dtype)

        if cfg.rand_init1:
            head1 = RN_Head(text_features, logit_scale).to(dtype)
        else:
            head1 = RN_Head(text_features, logit_scale).to(dtype)

        # To be optimized
        self.head = head
        self.head1 = head1

        # self.learn = DoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype)
        # self.learn = AdaptFormer(d_model=emb_dim, bottleneck=ffn_num, adapter_scalar=cfg.adapter_scalar, scalar_learnable=cfg.scalar_learnable,
                            #   adapter_layernorm_option=cfg.ln_opt, dtype=dtype)
        # self.learn = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim),
        #                           nn.ReLU(inplace=True),
        #                           nn.Linear(emb_dim, emb_dim)).to(dtype)
        self.learn = SSF(emb_dim, dtype=dtype)
