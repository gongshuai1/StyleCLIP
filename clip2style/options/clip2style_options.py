from argparse import ArgumentParser


class CLIP2StyleOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--dataset', default='ffhq', type=str,
                                 help='Type of dataset to train')

        self.parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')

        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')

        # Model weight
        self.parser.add_argument('--ir_se50_weights', default='../pretrained_models/model_ir_se50.pth', type=str,
                                 help="Path to facial recognition network used in ID loss")

        # Distributed Config
        self.parser.add_argument('--world_size', default=-1, type=int,
                                 help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=-1, type=int,
                                 help='node rank for distributed training')
        self.parser.add_argument('--dist_url', default='env://', type=str,
                                 help='url used to set up distributed training')
        self.parser.add_argument('--dist_backend', default='nccl', type=str,
                                 help='distributed backend')
        self.parser.add_argument('--gpu', default=None, type=int,
                                 help='GPU id to use')
        self.parser.add_argument('--multiprocessing_distributed', action='store_true',
                                 help='Use multi-processing distributed training to launch N processes per node, which has N GPUs.'
                                      'This is the fastest way to use PyTorch for '
                                      'either single node or multi node dataset parallel training')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
