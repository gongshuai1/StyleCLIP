import torch
import clip
import torch.nn as nn
from stylegan2.model import Generator, CLIP2Style


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class CLIP2StyleModel(nn.Module):
    def __init__(self, args):
        super(CLIP2StyleModel, self).__init__()
        self.args = args

        clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
        # CLIP image encoder
        self.image_encoder = clip_model.encode_image  # (3, 224, 224)
        # CLIP text encoder
        self.text_encoder = clip_model.encode_text
        # TODO: CLIP space: edit direction module
        # self.edit_direction = None
        # CLIP to StyleGAN space module - simplest implementation: MLP
        self.clip2style = CLIP2Style(clip_dim=512, style_dim=512, num_layers=6)
        # StyleGAN generator
        self.generator = Generator(self.args.stylegan_size, 512, 8)

        self.load_weight()
        self.configure_model()

    def frozen(self):
        # Freeze CLIP image encoder, CLIP text encoder, StyleGAN generator
        for param in self.image_encoder.parameters():
            param.requires_grad_(False)
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in self.generator.parameters():
            param.requires_grad_(False)

    def configure_model(self, freeze_edit=True, freeze_mapper=False):
        assert freeze_edit ^ freeze_mapper, \
            f'freeze_edit={freeze_edit} and freeze_mapper={freeze_mapper} must have one True and olly one True'
        self.frozen()

        # Freeze edit_direction module
        for param in self.edit_direction.parameters():
            param.requires_grad_(False if freeze_edit else True)

        # Freeze clip2style module
        for param in self.clip2style.parameters():
            param.requires_grad_(False if freeze_mapper else True)

    def load_weight(self):
        print(f'Loading StyleGAN weights from: {self.args.stylegan_weight}')
        styleGAN_ckpt = torch.load(self.args.stylegan_weight)
        self.generator.load_state_dict(styleGAN_ckpt['g_ema'], strict=False)

        if self.args.clip2style_checkpoint_path is not None:
            print(f'Loading from checkpoint: {self.args.clip2style_checkpoint_path}')
            ckpt = torch.load(self.args.clip2style_checkpoint_path)
            self.clip2style.load_state_dict(get_keys(ckpt, 'clip2style'), strict=True)
