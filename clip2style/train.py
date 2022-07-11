import random
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
from clip2style.options.clip2style_options import CLIP2StyleOptions
from clip2style.model.coach import Coach


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training.'
                      'This will turn on the CUDNN deterministic setting,'
                      'which can slow down your training considerably!'
                      'You may see unexpected behavior when restarting from checkpoints')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.This will completely disable dataset parallel.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngous_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size need to be adjusted accordingly
        args.world_size = ngous_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngous_per_node, args=(ngous_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngous_per_node, args)


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
            # For multiprocessing distributed training, rank needs to be global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method="env://127.0.0.1:23456",
                                world_size=args.world_size, rank=args.rank)

        coach = Coach(args)
        coach.clip2style_pretrain(args)


if __name__ == '__main__':
    args = CLIP2StyleOptions.parse()
    main(args)
