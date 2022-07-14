import torch
import os

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from clip2style.criteria.lpips.lpips import LPIPS
from clip2style.criteria.id_loss import IDLoss
from clip2style.criteria.moco_loss import MocoLoss
from clip2style.optimizers.ranger import Ranger
from clip2style.model.clip2style import CLIP2StyleModel


class Coach:
    def __init__(self, args):
        self.args = args

        self.global_step = 0

        self.net = CLIP2StyleModel(args)
        self.net.cuda(self.args.gpu)
        self.net = torch.nn.parallel.DistributedDataParallel(
            self.net, device_ids=[self.args.gpu], broadcast_buffers=False
        ).module

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

        # Initialize logger
        log_dir = os.path.join(args.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.args.save_interval is None:
            self.args.save_interval = self.args.max_steps

    def clip2style_configure_dataset(self):
        return [None]

    def clip2style_configure_optimizers(self):
        params = list(self.net.parameters())
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
