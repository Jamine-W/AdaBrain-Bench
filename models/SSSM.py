import math
import torch
import torch.nn as nn
from models.SGConv import GConv
from einops import rearrange, repeat


def swish(x):
    return x * torch.sigmoid(x)


class SNorm(nn.Module):
    def __init__(self, channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        norm = x.norm(dim=1, keepdim=True)
        rms = norm / (x.shape[1] ** 0.5)
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 device):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        self.device = torch.device(device)

        self.sn = RMSNorm(res_channels)

        self.S41 = GConv(d_model=2 * self.res_channels,
                         channels=4,
                         l_max=s4_lmax,
                         d_state=s4_d_state,
                         dropout=s4_dropout,
                         bidirectional=s4_bidirectional,
                         layer_norm=s4_layernorm)

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.attention = nn.MultiheadAttention(embed_dim=2 * self.res_channels, num_heads=4, dropout=s4_dropout,
                                               bias=True, batch_first=True)

        self.gelu = nn.GELU()

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)


        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)


    def generate_local_window_mask(self, seq_len, window_size):
        assert window_size % 2 == 1, "window_size shoule be odd number, like 7, 9, 11"

        half_window = window_size // 2

        mask = torch.full((seq_len, seq_len), float('-inf'))

        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = 0

        return mask

    def forward(self, input_data):
        x, original = input_data
        h = x
        B, C, L = x.shape
        x = self.sn(x)
        assert C == self.res_channels

        part_t = original.view([B, self.res_channels, L])
        h = h + part_t

        h = self.conv_layer(h)

        h = self.gelu(h)
        h_t, _ = self.S41(h)

        h_s = rearrange(h_t, "b c l -> b l c")
        SWA_mask = self.generate_local_window_mask(L, 1).to(
            device=self.device,
            dtype=h_s.dtype,
        )
        h_s, _ = self.attention(h_s, h_s, h_s, attn_mask=SWA_mask)
        h_s = rearrange(h_s, "b l c -> b c l")

        h = h_t + h_s

        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 device):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm,
                                                       device=device))

    def forward(self, input_data):
        noise = input_data
        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, noise))
            skip = skip_n + skip

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSM(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 codebook_size_t,
                 codebook_size_f,
                 if_codebook=True,
                 device='cpu'):
        super(SSSM, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels, out_channels, res_channels, s4_lmax // 19, 200)

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm,
                                             device=device)

        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))
        self.lm_head_t = nn.Linear(out_channels, codebook_size_t, bias=False)
        self.lm_head_f = nn.Linear(out_channels, codebook_size_f, bias=False)
        self.if_codebook = if_codebook
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, inputs, mask=None):
        bz, ch_num, seq_len, patch_size = inputs.shape
        inputs = self.patch_embedding(inputs, mask=mask)
        x = rearrange(inputs, 'b c s p -> b p (c s)')
        x = self.init_conv(x)
        x = self.residual_layer(x)
        x = self.final_conv(x)
        x = rearrange(x, 'b p (c s)  -> b c s p', p=patch_size, s=seq_len, c=ch_num)
        x = self.norm(x)
        if self.if_codebook:
            x = x.squeeze()[mask == 1]
            x_t = self.lm_head_t(x)
            x_f = self.lm_head_f(x)
            return (x_t, x_f)
        else:
            return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len, p_seq_len):
        super().__init__()
        self.d_model = p_seq_len
        self.p_seq_len = p_seq_len
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=(19, 7), stride=(1, 1),
                      padding=(9, 3),
                      groups=self.d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(p_seq_len), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, self.d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.p_seq_len)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3750):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(2), :].permute(1, 2, 0)
        return self.dropout(x)
