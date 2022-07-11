import torch

from torch import nn
from clip2style.criteria.lpips.lpips import LPIPS
from clip2style.criteria.id_loss import IDLoss
from clip2style.criteria.moco_loss import MocoLoss
from clip2style.optimizers.ranger import Ranger
from clip2style.model.clip2style import CLIP2StyleModel


class Coach:
    def __init__(self, args):
        self.args = args

        self.net = CLIP2StyleModel(args)

        # Loss
        #   lpips loss - reference - https://arxiv.org/abs/1801.03924
        #   irse loss - identify preservation
        #   L2 loss between origin image and reconstructed image
        if self.args.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.args.lpips_type)
        if self.args.id_lambda > 0:
            if 'ffhq' in self.args.dataset:
                self.id_loss = IDLoss(args)
            else:
                self.id_loss = MocoLoss()
        self.mse_loss = nn.MSELoss()

        # Optimizer
        self.optimizer = self.clip2style_configure_optimizers()

        # Dataset
        self.clip2style_train_dataset = self.clip2style_configure_dataset()

    def clip2style_configure_dataset(self):
        return [None]

    def clip2style_configure_optimizers(self):
        params = list(self.clip2style.parameters())
        if self.args.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.args.learning_rate)
        return optimizer

    def clip2style_pretrain(self, args):
        # Load dataset
        train_dataset = None

        # Encode image to latent code

        # Reconstruction latent code to image
        # Use pretrained StyleGAN generator

        # Compute loss:

        # Back forward

    def train(self, args):
        # Load dataset
        train_dataset = None

        # Optimizer

        # Encode image - CLIP image encoder

        # Encode text - CLIP text encoder

        # Get edit direction in CLIP space

        # Convert CLIP space to StyleGAN space

        # Generate image

        # Compute loss

        # Back forward
        pass
