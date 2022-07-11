import torch
import clip
import argparse
import os
import torchvision
import numpy as np
from PIL import Image
from models.stylegan2.model import Generator


def main(args):
    clip2style(args, is_mapper=True)


def clip2style(args, is_mapper):
    img_path = 'image/dv_224.png'
    img_orig = Image.open(img_path)
    img_orig = np.array(img_orig)
    img_orig = torch.from_numpy(img_orig)
    img_orig = img_orig[:, :, :3].permute(2, 0, 1)
    img_orig = torch.unsqueeze(img_orig, dim=0)

    clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    mean_latent = g_ema.mean_latent(4096)

    image_clip_space = clip_model.encode_image(img_orig)

    if is_mapper:
        img_gen_middle, latent_code_init, _ = g_ema([image_clip_space], return_latents=True,
                                                    truncation=args.truncation, truncation_latent=mean_latent)
        torchvision.utils.save_image(img_gen_middle.detach().cpu(),
                                     os.path.join(args.results_dir, "dv_middle_1024.jpg"),
                                     normalize=True, scale_each=True, range=(-1, 1))
        img_gen, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
    else:
        img_gen, _ = g_ema([image_clip_space], input_is_latent=True, randomize_noise=False,
                           input_is_stylespace=args.work_in_stylespace)

    # avg_pool = torch.nn.AvgPool2d(1024 // 224)
    # img_gen = avg_pool(img_gen)

    # final_result = torch.cat([img_orig, img_gen])
    final_result = img_gen
    torchvision.utils.save_image(final_result.detach().cpu(), os.path.join(args.results_dir, "dv_1024.jpg"),
                                 normalize=True, scale_each=True, range=(-1, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt",
                        help="pretrained StyleGAN2 weights")
    parser.add_argument('--work_in_stylespace', default=False, action='store_true')
    parser.add_argument("--results_dir", type=str, default="../clip2style/result")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="used only for the initial latent vector, and only when a latent code path is"
                             "not provided")
    args = parser.parse_args()

    main(args)
