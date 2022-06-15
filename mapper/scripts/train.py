"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import random
import warnings
import builtins

sys.path.append(".")
sys.path.append("..")
# sys.path.append("~/pythonProject/StyleCLIP")

from mapper.options.train_options import TrainOptions
from mapper.training.coach import Coach
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp


def main(opts):
    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    # pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    if opts.seed is not None:
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed traning.'
                      'This will turn on the CUDNN deterministic setting,'
                      'which can slow down your training considerably!'
                      'You may see unexpected behavior when restarting from checkpoints')

    if opts.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable dataset parallel.')

    if opts.dist_url == 'env://' and opts.world_size == -1:
        opts.world_size = int(os.environ['WORLD_SIZE'])

    opts.distributed = opts.world_size > 1 or opts.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if opts.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size need to be adjusted accordingly
        opts.world_size = ngpus_per_node * opts.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opts))
    else:
        # Simply call main_worker function
        main_worker(opts.gpu, ngpus_per_node, opts)


def main_worker(gpu, ngpus_per_node, args):
    torch.cuda.set_device(gpu)
    args.gpu = gpu

    # Suppress printing if not master
    # if args.multiprocessing_distributed and args.gpu != 0:
    #     def print_pass(*args):
    #         pass
    #
    #     builtins.print = print_pass

    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the golbal rank among all the process
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method="env://127.0.0.1:23456",
                                world_size=args.world_size, rank=args.rank)

        coach = Coach(args)
        coach.train()


if __name__ == '__main__':
    opts = TrainOptions().parse()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23456'
    main(opts)
