import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.utils import get_data, ExperimentDataset
from tqdm import tqdm
from src.utils import calc_spec, deflection_biexp_calc
from train import weighted_mse_loss, weighted_mean, sigmoid_loss
import argparse
import yaml
import torchvision
from torch.optim import Adam
from collections import namedtuple
import os
import numpy as np

# Define the DoubleConv block
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

class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
        
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            if self.attn:
                # Attention block of Unet
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
            
        # Downsample
        out = self.down_sample_conv(out)
        return out
    

class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
             for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
    
    def forward(self, x, t_emb=None, context=None):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
                
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        
        return out
    

class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)
        
        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Self Attention
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        return out


class VQVAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']
        
        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.codebook_size = model_config['codebook_size']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        
        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        
        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)
        ####################################################
        
        ##################### Decoder ######################
        
        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i-1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
    
    def quantize(self, x):
        B, C, H, W = x.shape
        
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        
        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        
        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()
        
        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)
        
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out
    
    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses
    

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # Imagnet normalization for (0-1)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
    
    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out
    

class Discriminator(nn.Module):
    r"""
    PatchGAN Discriminator.
    Rather than taking IMG_CHANNELSxIMG_HxIMG_W all the way to
    1 scalar value , we instead predict grid of values.
    Where each grid is prediction of how likely
    the discriminator thinks that the image patch corresponding
    to the grid cell is real
    """
    
    def __init__(self, im_channels=3,
                 conv_channels=[64, 128, 256],
                 kernels=[4,4,4,4],
                 strides=[2,2,2,1],
                 paddings=[1,1,1,1]):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i !=0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # Load pretrained vgg model from torchvision
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        # Freeze vgg model
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):
        # Return output of vgg features
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, net='vgg', version='0.1', use_dropout=True):
        super(LPIPS, self).__init__()
        self.version = version
        # Imagenet normalization
        self.scaling_layer = ScalingLayer()
        ########################
        
        # Instantiate vgg model
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(pretrained=True, requires_grad=False)
        
        # Add 1x1 convolutional Layers
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.ModuleList(self.lins)
        ########################
        
        # Load the weights of trained LPIPS model
        import inspect
        import os
        model_path = os.path.abspath(
            os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
        print('Loading model from: %s' % model_path)
        self.load_state_dict(torch.load(
            '../StableDiffusion-PyTorch/models/weights/v0.1/vgg.pth', map_location=device), strict=False)
        ########################
        
        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        ########################
    
    def forward(self, in0, in1, normalize=False):
        # Scale the inputs to -1 to +1 range if needed
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        ########################
        
        # Normalize the inputs according to imagenet normalization
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        ########################
        
        # Get VGG outputs for image0 and image1
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        ########################
        
        # Compute Square of Difference for each layer output
        for kk in range(self.L):
            feats0[kk], feats1[kk] = torch.nn.functional.normalize(outs0[kk], dim=1), torch.nn.functional.normalize(
                outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        ########################
        
        # 1x1 convolution followed by spatial average on the square differences
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        val = 0
        
        # Aggregate the results of each layer
        for l in range(self.L):
            val += res[l]
        return val


def make_latent_dataset(vqvae, dataloader, save_path):
    vqvae.eval()
    quantized_latents = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images = data['image'].to(next(vqvae.parameters()).device)
            images = images.view(images.size(0), 1, images.size(2), images.size(3))
            z = vqvae.encoder(images)
            z = vqvae.pre_quant(z)
            z = vqvae.pre_quant2(z)
            quantized, _, _ = vqvae.vq(z)  # Get the quantized variables
            quantized_latents.append(quantized.cpu())
    
    quantized_latents = torch.cat(quantized_latents, dim=0)
    
    # Save the quantized latent dataset to disk
    torch.save(quantized_latents, save_path)
    print(f"Quantized latent dataset saved to {save_path}")


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "vqvae"
    args.epochs = 100
    args.noise_steps = 1000
    args.phys = True  # For PIDDIM, change to True
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 4
    args.image_height = 256
    args.image_width = 512
    args.real_size = (256, 512)
    args.features = ["E", "P", "ms"]
    args.dataset_path = r"data/with_gain"
    args.csv_path = "data/params.csv"
    args.device = "cuda:1" if torch.cuda.is_available() else "cpu"  # Adjust for available devices
    args.lr = 1e-3
    args.exclude = []  # ['train/19']
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_settings = [32., 15., 15.]
    args.sample_size = 8
    args.electron_pointing_pixel = 62
    args.seed = 42
    args.split = True

    device = args.device


    config_path = '../StableDiffusion-PyTorch/config/mnist.yaml'
    config = load_config(config_path)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    train_config['autoencoder_img_save_steps'] = 1024
    train_config['perceptual_weight'] = 1 #0.5
    train_config['disc_weight'] = 1 #0.5

    train_loader, valid_loader = get_data(args)
    config['autoencoder_params']['num_down_layers'] = 1
    config['autoencoder_params']['num_up_layers'] = 1
    model = VQVAE(im_channels=12, model_config=config['autoencoder_params']).to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()

    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=12).to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    disc_step_start = train_config['disc_start']
    step_count = 0

    acc_steps = 4
image_save_steps = train_config['autoencoder_img_save_steps']
img_save_count = 0

for epoch_idx in range(50):
    recon_losses = []
    codebook_losses = []
    #commitment_losses = []
    perceptual_losses = []
    disc_losses = []
    gen_losses = []
    losses = []

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

    for im in tqdm(train_loader):
        step_count += 1
        im = im['s2_target'].float().to(device)

        # Fetch autoencoders output(reconstructions)
        model_output = model(im)
        output, z, quantize_losses = model_output

        ######### Optimize Generator ##########
        # L2 Loss
        recon_loss = recon_criterion(output, im) 
        recon_losses.append(recon_loss.item())
        recon_loss = recon_loss / acc_steps
        g_loss = (recon_loss +
                  (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                  (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
        codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
        # Adversarial loss only if disc_step_start steps passed
        if step_count > disc_step_start:
            disc_fake_pred = discriminator(model_output[0])
            disc_fake_loss = disc_criterion(disc_fake_pred,
                                            torch.ones(disc_fake_pred.shape,
                                                       device=disc_fake_pred.device))
            gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
            g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
        lpips_loss = torch.mean(lpips_model(output[:, [3,2,1], :, :], im[:, [3,2,1], :, :])) / acc_steps
        perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
        g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps
        losses.append(g_loss.item())
        g_loss.backward()
        #####################################

        ######### Optimize Discriminator #######
        if step_count > disc_step_start:
            fake = output
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(im)
            disc_fake_loss = disc_criterion(disc_fake_pred,
                                            torch.zeros(disc_fake_pred.shape,
                                                        device=disc_fake_pred.device))
            disc_real_loss = disc_criterion(disc_real_pred,
                                            torch.ones(disc_real_pred.shape,
                                                       device=disc_real_pred.device))
            disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
            disc_losses.append(disc_loss.item())
            disc_loss = disc_loss / acc_steps
            disc_loss.backward()
            if step_count % acc_steps == 0:
                optimizer_d.step()
                optimizer_d.zero_grad()
        #####################################

        if step_count % acc_steps == 0:
            optimizer_g.step()
            optimizer_g.zero_grad()
    optimizer_d.step()
    optimizer_d.zero_grad()
    optimizer_g.step()
    optimizer_g.zero_grad()
    if len(disc_losses) > 0:
        print(
            'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
            'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
            format(epoch_idx + 1,
                   np.mean(recon_losses),
                   np.mean(perceptual_losses),
                   np.mean(codebook_losses),
                   np.mean(gen_losses),
                   np.mean(disc_losses)))
    else:
        print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
              format(epoch_idx + 1,
                     np.mean(recon_losses),
                     np.mean(perceptual_losses),
                     np.mean(codebook_losses)))

    print(model.encoder_conv_in.weight.shape)
    torch.save(model.state_dict(), './model_weights/vqvae.pth')
    torch.save(discriminator.state_dict(), './model_weights/discriminator.pth')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "vqvae"
    args.epochs = 100
    args.noise_steps = 1000
    args.phys = True  # For PIDDIM, change to True
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 4
    args.image_height = 256
    args.image_width = 512
    args.real_size = (256, 512)
    args.features = ["E", "P", "ms"]
    args.dataset_path = r"data/with_gain"
    args.csv_path = "data/params.csv"
    args.device = "cuda:1" if torch.cuda.is_available() else "cpu"  # Adjust for available devices
    args.lr = 1e-3
    args.exclude = []  # ['train/19']
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_settings = [32., 15., 15.]
    args.sample_size = 8
    args.electron_pointing_pixel = 62
    args.seed = 42
    args.split = True

    # Load data
    train_loader, valid_loader = get_data(args)
    best_val_loss = 1e12

    el_pointing_adjusted = int(args.electron_pointing_pixel/(args.real_size[1]/args.image_width))
    pixel_in_mm_adjusted = 0.137*(args.real_size[1]/args.image_width)
    fing_x = int(8/(args.real_size[1]/args.image_width))
    fing_y = int(8/(args.real_size[0]/args.image_height))

    deflection_MeV, deflection_MeV_dx = deflection_biexp_calc(args.batch_size, args.real_size[1], el_pointing_adjusted, pixel_in_mm_adjusted)
    deflection_MeV = deflection_MeV.to(args.device)

    device = args.device
    # Initialize model and optimizer
    model = VQVAE().to(args.device)  # Move model to the specified device
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    epochs = args.epochs
    for epoch in tqdm(range(epochs)):
        model.train()
        running_mse = 0
        running_vq = 0
        running_phys1 = 0
        running_phys2 = 0
        for data in train_loader:
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            acq_time = settings[:, 2]
            # Move data to the specified device
            # Ensure data is in the correct shape, e.g., [batch_size, 1, height, width]
            images = images.view(images.size(0), 1, args.image_height, args.image_width)
            recon, vq_loss = model(images)
            mse_loss = F.mse_loss(recon, images)

            if args.phys:
                _, x_t_spectr = calc_spec(((images.clamp(-1, 1) + 1) / 2).to(args.device), 
                                            args.electron_pointing_pixel, 
                                            deflection_MeV, 
                                            acquisition_time_ms=acq_time, 
                                            resize=args.real_size,
                                            image_gain=0,
                                            device=args.device,
                                            deflection_MeV_dx=None)
                _, pred_spectr = calc_spec(((recon.clamp(-1, 1) + 1) / 2).to(args.device), 
                                            args.electron_pointing_pixel, 
                                            deflection_MeV, 
                                            acquisition_time_ms=acq_time, 
                                            resize=args.real_size,
                                            image_gain=0,
                                            device=args.device,
                                            deflection_MeV_dx=None)
                concatenated = torch.cat((x_t_spectr, pred_spectr), dim=-1)
                max_val = torch.max(concatenated)
                min_val = torch.min(concatenated)
                x_t_spectr_norm = (x_t_spectr - min_val) / ((max_val - min_val) / 2) - 1
                pred_spectr_norm = (pred_spectr - min_val) / ((max_val - min_val) / 2) - 1
                pred_norm = (recon.clamp(-1, 1) + 1) / 2
                pred_norm[:, :, :fing_y, :fing_x] = 0
                phys_weight = 0.1
                phys1 = weighted_mse_loss(x_t_spectr_norm, pred_spectr_norm, phys_weight)
                phys2 = weighted_mean(sigmoid_loss(pred_norm, el_pointing=el_pointing_adjusted, pixel_in_mm=pixel_in_mm_adjusted, device=device), phys_weight)
                # phys2 = torch.Tensor([0]).to(device)
            else:
                phys1 = torch.Tensor([0]).to(device)
                phys2 = torch.Tensor([0]).to(device)

            loss = mse_loss + vq_loss + phys1 + phys2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients for next batch
            running_mse += mse_loss.item()
            running_vq += vq_loss.item()
            running_phys1 += phys1.item()
            running_phys2 += phys2.item()
        print(f'Epoch: {epoch + 1}, MSE_loss: {running_mse / len(train_loader):.4f}, vq_loss: {running_vq / len(train_loader):.4f}, phys1_loss: {running_phys1 / len(train_loader):.4f}, phys2_loss: {running_phys2 / len(train_loader):.4f}')

    if args.phys:
        name = 'final_128_vqvae_model_phys.pth'
    else:
        name = 'final_128_vqvae_model.pth'
    torch.save(model.state_dict(), name)

    print("Training complete and model saved.")
