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

src_image = '/home/ubuntu/data/psp/frame/Trump_cl_aligned/result.png'
dst_image = '/home/ubuntu/data/psp/frame/Trump_cl_aligned/Trump_GoogleScrape2_099_0.png'
exp_dir = '/home/ubuntu/data/psp/output/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq_inter_hq_trump/2'

# src_image = '/media/ubuntu/Data1/data/Trump/WholeFace/_CustomBatches/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq/00278.png'
# dst_image = '/media/ubuntu/Data1/data/Trump/WholeFace/_CustomBatches/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq/Trump_GoogleScrape2_159_0.png'
# exp_dir = '/home/ubuntu/data/psp/output/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq_inter_hq_trump/3'


test_batch_size = 1
test_workers = 1
checkpoint_path = '/home/ubuntu/data/psp/model/trump_encoder/checkpoints/best_model.pt'
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

# print(a.shape)
# print(b.shape)

# input_is_encode is used to stop transforming input_code by style layer
# That transformation is only needed for random input vectors
# rec_src, rec_v_src = net(v_src.to("cuda"),
#                  input_code=True,
#                  return_latents=True,
#                  input_is_encode=True) 

# rec_dst, rec_v_dst = net(v_dst.to("cuda"),
#                  input_code=True,
#                  return_latents=True,
#                  input_is_encode=True) 



if not os.path.exists(opts.exp_dir):
    os.makedirs(opts.exp_dir)


step = (v_dst - v_src) / num_step
for s in range(num_step + 1):
    v = v_src + step * s
    output, _ = net(v.to("cuda"),
                     resize = False,
                     input_code=True,
                     return_latents=True,
                     input_is_encode=True,
                     randomize_noise=False)
    print('----------------------------')
    print(output.shape)
    Image.fromarray(np.array(tensor2im(output[0]))).save(
        os.path.join(opts.exp_dir, str(s).zfill(5) + '.png'))

# Image.fromarray(np.array(tensor2im(rec_src[0]))).save('rec_src.png')
# Image.fromarray(np.array(tensor2im(rec_dst[0]))).save('rec_dst.png')

# print(v_src - rec_v_src)
# print(v_dst - rec_v_dst)