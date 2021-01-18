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


src_image = '/home/ubuntu/data/psp/frame/Trump_cl_aligned/result.png'
exp_dir = '/home/ubuntu/data/psp/output/Trump_cl_aligned_frontal'

test_batch_size = 1
test_workers = 1
checkpoint_path = 'pretrained_models/psp_ffhq_frontalization.pt'
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

net = pSp(opts)
net.eval()
net.cuda()

a, v_src = net(src_im.unsqueeze(0).to("cuda").float(),
            return_latents=True,
            resize = False,
            randomize_noise = False)

if not os.path.exists(opts.exp_dir):
    os.makedirs(opts.exp_dir)

Image.fromarray(np.array(tensor2im(a[0]))).save(
    os.path.join(opts.exp_dir, 'result.png'))

