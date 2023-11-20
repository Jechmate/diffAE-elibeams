
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple
import math


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, height, width):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.height * self.width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.height, self.width)


def pos_encoding(timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) *
                    torch.arange(start=0, end=half, dtype=torch.float32) /
                    half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
            # nn.Dropout(p=0.2),
        )

        self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, 2 * in_channels),
            )
        self.cond_emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, in_channels),
            )
    
    def scale_shift(self, x, emb, cond, scale_bias=1):
        if emb is not None:
            while len(emb.shape) < len(x.shape):
                emb = emb[..., None]
        while len(cond.shape) < len(x.shape):
            cond = cond[..., None]
        scale_shifts = [emb, cond]

        for i, each in enumerate(scale_shifts):
            if each is None:
                a = None
                b = None
            else:
                if each.shape[1] == self.in_channels * 2:
                    a, b = torch.chunk(each, 2, dim=1)
                else:
                    a = each
                    b = None
            scale_shifts[i] = (a, b)

        biases = [scale_bias] * len(scale_shifts)
        x = self.maxpool_conv[0](x)
        for i, (scale, shift) in enumerate(scale_shifts):
            if scale is not None:
                x = x * (biases[i] + scale)
                if shift is not None:
                    x = x + shift
        x = self.maxpool_conv[1:](x)
        return x

    def forward(self, x, t, cond):
        if t is not None:
            emb_out = self.emb_layers(t)
        else:
            emb_out = None
        if cond is not None:
            cond_out = self.cond_emb_layers(cond)
        else:
            cond_out = None
        if cond_out is not None:
                    while len(cond_out.shape) < len(x.shape):
                        cond_out = cond_out[..., None]
        x = self.scale_shift(x, emb=emb_out, cond=cond_out)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, 2 * out_channels),
            )
        self.cond_emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, 2 * out_channels), # TODO why is this x2 necessary?
            )

        self.skip_connection = nn.Conv2d(
                                    in_channels,
                                    out_channels,
                                    1,
                                    padding=0)

    def scale_shift(self, x, emb, cond, scale_bias=1):
        if emb is not None:
            while len(emb.shape) < len(x.shape):
                emb = emb[..., None]
        while len(cond.shape) < len(x.shape):
            cond = cond[..., None]
        scale_shifts = [emb, cond]

        for i, each in enumerate(scale_shifts):
            if each is None:
                a = None
                b = None
            else:
                if each.shape[1] == self.in_channels * 2:
                    a, b = torch.chunk(each, 2, dim=1)
                else:
                    a = each
                    b = None
            scale_shifts[i] = (a, b)

        biases = [scale_bias] * len(scale_shifts)
        x = self.conv[0](x)
        for i, (scale, shift) in enumerate(scale_shifts):
            if scale is not None:
                x = x * (biases[i] + scale)
                if shift is not None:
                    x = x + shift
        x = self.conv[1](x)
        return x

    def forward(self, x, skip_x, t, cond):
        h = self.skip_connection(skip_x)
        x = self.up(x)
        if t is not None:
            emb_out = self.emb_layers(t)
        else:
            emb_out = None
        if cond is not None:
            cond_out = self.cond_emb_layers(cond)
        else:
            cond_out = None
        if cond_out is not None:
                    while len(cond_out.shape) < len(x.shape):
                        cond_out = cond_out[..., None]
        x = self.scale_shift(x, emb=emb_out, cond=cond_out)
        return h + x


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda", img_width=256, img_height=128, feat_num=5):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.cond_dim = feat_num
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention(128, img_height//2, img_width//2)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention(256, img_height//4, img_width//4)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, img_height//8, img_width//8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(256, 128, emb_dim=time_dim)
        self.sa4 = SelfAttention(128, img_height//4, img_width//4)
        self.up2 = Up(128, 64, emb_dim=time_dim)
        self.sa5 = SelfAttention(64, img_height//2, img_width//2)
        self.up3 = Up(64, 64, emb_dim=time_dim)
        self.sa6 = SelfAttention(64, img_height, img_width)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=64,
            time_out_channels=self.time_dim,
        )

    def forward(self, x, t, y, t_cond=None):
        if t_cond is None:
            t_cond = t
        if t is not None:
            _t_emb = pos_encoding(t, 64)
            _t_cond_emb = pos_encoding(t_cond, 64)
        res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=y,
                time_cond_emb=_t_cond_emb,
            )
        t = res.time_emb
        y = res.emb
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t, y)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t, y)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t, y)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t, y)
        x = self.sa4(x)
        x = self.up2(x, x2, t, y)
        x = self.sa5(x)
        x = self.up3(x, x1, t, y)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class SemEncoder(nn.Module):
    def __init__(self, c_in=1, time_dim=256, device="cuda", img_width=256, img_height=128, feat_num=5, latent_dim=128):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.latent_dim = latent_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention(128, img_height//2, img_width//2)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention(256, img_height//4, img_width//4)
        self.down3 = Down(256, 512, emb_dim=time_dim)
        self.sa3 = SelfAttention(512, img_height//8, img_width//8)
        self.down4 = Down(512, 512, emb_dim=time_dim)
        self.sa4 = SelfAttention(512, img_height//16, img_width//16)
        self.out = nn.Sequential(
                nn.GroupNorm(1, 512),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(512, latent_dim, kernel_size=1),
                nn.Flatten(),
            )
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=feat_num,
            time_out_channels=self.time_dim,
        )
    

    def forward(self, x, y):
        res = self.time_embed.forward(
                time_emb=y,
                cond=None,
                time_cond_emb=None,
            )
        y = res.time_emb
        x1 = self.inc(x)
        x2 = self.down1(x1, None, y)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, None, y)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, None, y)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, None, y)
        x5 = self.sa4(x5)
        output = self.out(x5)
        return output


# Taken directly from diffAE paper
class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, feat_num=3, device='cuda'):
        super().__init__()
        self.num_channels = 128 # latent_dim
        self.num_time_emb_channels = 32
        self.num_time_layers = 2
        self.time_last_act = False
        self.num_layers = 20
        self.num_hid_channels = 1024
        self.use_norm = False
        self.dropout = 0
        self.skip_layers = [] # skip should be added everywhere according to the paper but nowhere according to config
        self.condition_bias = 0
        self.device = device

        self.label_prep = nn.Sequential(
            nn.BatchNorm1d(feat_num),
            nn.Linear(feat_num, self.num_channels),
            nn.SiLU(),
        ).to(device)

        layers = []
        for i in range(self.num_time_layers):
            if i == 0:
                a = self.num_time_emb_channels
                b = self.num_channels
            else:
                a = self.num_channels
                b = self.num_channels
            layers.append(nn.Linear(a, b))
            if i < self.num_time_layers - 1 or self.time_last_act:
                layers.append(nn.SiLU())
        self.time_embed = nn.Sequential(*layers).to(device)

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            if i == 0:
                act = nn.SiLU()
                norm = self.use_norm
                cond = True
                a, b = self.num_channels, self.num_hid_channels
                dropout = self.dropout
            elif i == self.num_layers - 1:
                act = nn.Identity()
                norm = False
                cond = False
                a, b = self.num_hid_channels, self.num_channels
                dropout = 0
            else:
                act = nn.SiLU()
                norm = self.use_norm
                cond = True
                a, b = self.num_hid_channels, self.num_hid_channels
                dropout = self.dropout

            if i in self.skip_layers:
                a += self.num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=self.num_channels,
                    use_cond=cond,
                    condition_bias=self.condition_bias,
                    dropout=dropout,
                    device=self.device
                ))
        self.last_act = nn.Identity()

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.num_time_emb_channels)
        cond = self.time_embed(t)
        y = self.label_prep(y).squeeze()
        cond += y
        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                # h = torch.cat([h, x], dim=1)
                pass # Do they use the skip connections or not? according to paper yes, according to code no
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return h
    
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(self.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation,
        use_cond: bool,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
        device = 'cuda'
    ):
        super().__init__()
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels).to(device)
        self.act = activation.to(device)
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels).to(device)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb).to(device)
        if norm:
            self.norm = nn.LayerNorm(out_channels).to(device)
        else:
            self.norm = nn.Identity().to(device)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout).to(device)
        else:
            self.dropout = nn.Identity().to(device)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class EmbedReturn(NamedTuple):
    # style and time
    emb: torch.Tensor = None
    # time only
    time_emb: torch.Tensor = None
    # style only (but could depend on time)
    style: torch.Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_channels, time_out_channels),
            nn.SiLU(),
            nn.Linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)