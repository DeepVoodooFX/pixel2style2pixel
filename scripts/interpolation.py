import os
import sys
from argparse import Namespace

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import sys
sys.path.append(".")
sys.path.append("..")

from models.psp import pSp
from utils.common import tensor2im, log_input_image

# src_image = '/home/ubuntu/data/psp/frame/inversion_images/benedict_cumberbatch.jpg'
# dst_image = '/home/ubuntu/data/psp/frame/inversion_images/benedict_cumberbatch_flip.jpg'
# exp_dir = '/home/ubuntu/data/psp/output/inversion_images_style_mixed'

# src_image = '/home/ubuntu/data/psp/frame/Trump_cl_aligned/donal_trump.jpg'
# dst_image = '/home/ubuntu/data/psp/frame/Trump_cl_aligned/donal_trump_flip.jpg'
# exp_dir = '/home/ubuntu/data/psp/output/Trump_cl_aligned_inter'

# src_image = '/home/ubuntu/data/psp/frame/CelebA_HQ/00462.jpg'
# dst_image = '/home/ubuntu/data/psp/frame/CelebA_HQ/00462_flip.jpg'
# exp_dir = '/home/ubuntu/data/psp/output/CelebA_HQ_inter'

src_image = '/media/ubuntu/Data1/dfl/data/interpolation/00004.png'
dst_image = '/media/ubuntu/Data1/dfl/data/interpolation/00005.png'
exp_dir = '/media/ubuntu/Data1/dfl/data/interpolation_results'

# src_image = '/media/ubuntu/Data1/data/Trump/WholeFace/_CustomBatches/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq/00278.png'
# dst_image = '/media/ubuntu/Data1/data/Trump/WholeFace/_CustomBatches/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq/Trump_GoogleScrape2_159_0.png'
# exp_dir = '/home/ubuntu/data/psp/output/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq_inter_hq_trump/3'


test_batch_size = 1
test_workers = 1
checkpoint_path = '/media/ubuntu/Data1/dfl/model/psp/best_model_hd_trump.pt'
num_step = 100

transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

ckpt = torch.load(checkpoint_path, map_location='cpu')
opts = ckpt['opts']
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False
opts = Namespace(**opts)
opts.test_batch_size = test_batch_size
opts.test_workers = test_workers
opts.exp_dir = exp_dir
opts.checkpoint_path = checkpoint_path


src_im = Image.open(src_image)
src_im = src_im.convert('RGB') if opts.label_nc == 0 else src_im.convert('L')
src_im = transform(src_im)

dst_im = Image.open(dst_image)
dst_im = dst_im.convert('RGB') if opts.label_nc == 0 else dst_im.convert('L')
dst_im = transform(dst_im)

net = pSp(opts)
net.eval()
net.cuda()

a, v_src = net(src_im.unsqueeze(0).to("cuda").float(),
            return_latents=True,
            resize = False,
            randomize_noise = False)


b, v_dst = net(dst_im.unsqueeze(0).to("cuda").float(),
            return_latents=True,
            resize = False,
            randomize_noise = False)

if not os.path.exists(opts.exp_dir):
    os.makedirs(opts.exp_dir)


step = (v_dst - v_src) / num_step
for s in range(num_step + 1):
    v = v_src + step * s

    # Save latent vectors
    np.save(os.path.join(opts.exp_dir, str(s).zfill(5) + '.npy'),
            v.cpu().detach().numpy())
    
    # # Test saved latent vectors
    # v = torch.from_numpy(
    #     np.load(os.path.join(opts.exp_dir, str(s).zfill(5) + '.npy'))).to(device='cuda')

    # Decode latent vectors using stylegan2
    output, _ = net(v.to("cuda"),
                     resize = False,
                     input_code=True,
                     return_latents=True,
                     input_is_encode=True,
                     randomize_noise=False)
    # Save decoded image
    Image.fromarray(np.array(tensor2im(output[0]))).save(
        os.path.join(opts.exp_dir, str(s).zfill(5) + '.png'))

    print('Step {}/{}'.format(s + 1, num_step))
