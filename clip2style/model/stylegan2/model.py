import torch
import math
import random
from torch import nn
from torch.nn import functional as F

from fused_act import FusedLeakyReLU, fused_leaky_relu
from upfirdn2d import upfirdn2d


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()
    return k


class UpSample(nn.Module):
    def __init__(self, kernel, factor=2):
        super(UpSample, self).__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor -1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class DownSample(nn.Module):
    def __init__(self, kernel, factor=2):
        super(DownSample, self).__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    
    
class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, up_sample_factor=1):
        super(Blur, self).__init__()
        kernel = make_kernel(kernel)

        if up_sample_factor > 1:
            kernel = kernel * (up_sample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, self.pad)
        return out


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        self.input = nn.Parameter(torch.randn((1, channel, size, size)))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            up_sample=False,
            down_sample=False,
            blur_kernel=[1, 3, 3, 1]
    ):
        super(ModulatedConv2d, self).__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up_sample = up_sample
        self.down_sample = down_sample

        if up_sample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), up_sample_factor=factor)

        if down_sample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_channel = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_channel)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style, input_in_stylespace=False):
        batch, in_channel, height, width = input.shape

        if not input_in_stylespace:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, 1, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.up_sample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=self.padding, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.down_sample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out, style


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            up_sample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True
    ):
        super(StyledConv, self).__init__()
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            up_sample=up_sample,
            blur_kernel=blur_kernel,
            demodulate=demodulate
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, input_is_stylespace=False):
        out, style = self.conv(input, style, input_is_stylespace=input_is_stylespace)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out, style


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, up_sample=True, blur_kernel=[1, 3, 3, 1]):
        super(ToRGB, self).__init__()
        if up_sample:
            self.up_sample = UpSample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None, input_is_stylespace=False):
        out, style = self.conv(input, style, input_is_stylespace=input_is_stylespace)
        out = out + self.bias

        if skip is not None:
            skip = self.up_sample(skip)
            out = out + skip

        return out, style


class Generator(nn.Module):
    def __init__(
            self,
            image_size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01
    ):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, up_sample=False)

        self.log_size = int(math.log(image_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel, out_channel, 3, style_dim, up_sample=True, blur_kernel=blur_kernel
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(
                ToRGB(out_channel, style_dim)
            )
            in_channel = out_channel

        self.n_latent = self.log_size ** 2 -2

    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        mean_latent = self.style(latent_in).mean(0, keepdim=True)
        return mean_latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            input_is_stylespace=False,
            noise=None,
            randomize_noise=True
    ):
        if not input_is_latent and not input_is_stylespace:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1 and not input_is_stylespace:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t

        if input_is_stylespace:
            latent = styles[0]
        elif len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)

        style_vector = []

        if not input_is_stylespace:
            out = self.input(latent)
            out, out_style = self.conv1(out, latent[:, 0], noise=noise[0])
            style_vector.append(out_style)

            skip, out_style = self.to_rgb1(out, latent[:, 1])
            style_vector.append(out_style)

            i = 1
        else:
            out = self.input(latent[0])
            out, out_style = self.conv1(out, latent[0], noise=noise[0], input_is_stylespace=input_is_stylespace)
            style_vector.append(out_style)

            skip, out_style = self.to_rgb1(out, latent[1], input_is_stylespace)
            style_vector.append(out_style)

            i = 2

        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            if not input_is_stylespace:
                out, out_style1 = conv1(out, latent[:, i], noise=noise1)
                out, out_style2 = conv2(out, latent[:, i + 1], noise=noise2)
                skip, rgb_style = to_rgb(out, latent[:, i + 2], skip)

                style_vector.extend([out_style1, out_style2, rgb_style])
                i += 2
            else:
                out, out_style1 = conv1(out, latent[i], noise=noise1, input_is_stylespace=input_is_stylespace)
                out, out_style2 = conv2(out, latent[i + 1], noise=noise2, input_is_stylespace=input_is_stylespace)
                skip, rgb_style = to_rgb(out, latent[i + 2], skip, input_is_stylespace=input_is_stylespace)

                style_vector.extend([out_style1, out_style2, rgb_style])
                i += 3

        image = skip

        if return_latents:
            return image, latent, style_vector
        else:
            return image, None


class CLIP2Style(nn.Module):
    def __init__(self, clip_dim, style_dim, num_layers=6, lr_mlp=0.01):
        super(CLIP2Style, self).__init__()
        self.clip_dim = clip_dim
        self.style_dim = style_dim
        self.num_layers = num_layers

        layers = [EqualLinear(clip_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu')]

        for i in range(1, num_layers):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.clip2style = nn.Sequential(*layers)

    def forward(self, input):
        batch, clip_dim = input.shape
        assert clip_dim == self.clip_dim
        out = self.clip2style(input)
        return out