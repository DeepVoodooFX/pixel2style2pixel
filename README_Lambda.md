Install
===

__Setup Environment__

```
conda env create --file=environment/environments.yml
conda activate psp_env

cd
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python setup.py install
```

__Prepare Data__

Download pretrained model for encoder, put it under `pretrained_models`

```
https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view
```

__Download landmark files__

```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```

Run
===

```
python scripts/inference_recon.py \
--exp_dir=/home/ubuntu/data/psp/output/Trump_aligned_bf_recon_finetuned2500 \
--checkpoint_path=/home/ubuntu/data/psp/model/trump_encoder/checkpoints/best_model.pt \
--data_path=/home/ubuntu/data/psp/frame/Trump_aligned_bf \
--test_batch_size=8 \
--test_workers=2 \
--couple_outputs


python scripts/inference_recon.py \
--exp_dir=/home/ubuntu/data/psp/output/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq_recon_finetuned22500 \
--checkpoint_path=/home/ubuntu/data/psp/model/trump_encoder/checkpoints/best_model.pt \
--data_path=/media/ubuntu/Data1/data/Trump/WholeFace/_CustomBatches/TomsSelect+AllGetty+AllGoogle_mini_dfl2ffhq \
--test_batch_size=8 \
--test_workers=2 \
--couple_outputs



python scripts/inference_recon_dfl.py \
--exp_dir=/home/ubuntu/data/psp/output/10_aligned_wf_dfl \
--checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=/home/ubuntu/data/psp/frame/images1024x1024_10 \
--aligned_path=/home/ubuntu/data/psp/frame/10_aligned_wf \
--test_batch_size=1 --test_workers=1 --couple_outputs



python scripts/inference_recon_dfl.py \
--exp_dir=/home/ubuntu/data/psp/output/Trump_dfl \
--checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=/home/ubuntu/data/psp/frame/Trump \
--aligned_path=/home/ubuntu/data/psp/frame/Trump_aligned_wf \
--test_batch_size=1 --test_workers=1 --couple_outputs


python scripts/style_mixing.py \
--exp_dir=/home/ubuntu/data/psp/output/inversion_images_style_mixed \
--checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=/home/ubuntu/data/psp/frame/inversion_images \
--test_batch_size=4 \
--test_workers=4 \
--n_outputs_to_generate=5 \
--latent_mask=8,9,10,11,12,13,14,15,16,17


python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=/home/ubuntu/data/psp/model/trump_encoder \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=2500 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1 \
--checkpoint_path=/home/ubuntu/pixel2style2pixel/pretrained_models/best_model_2500.pt

```


Misc
===

```

python main.py \
videoed extract-video \
--input-file=/home/ubuntu/data/psp/video/Trump.mp4 \
--output-dir=/home/ubuntu/data/psp/frame/Trump \
--output-ext=png --fps=0



python main.py extract --detector s3fd \
--input-dir /home/ubuntu/data/psp/frame/Trump \
--output-dir /home/ubuntu/data/psp/frame/Trump_aligned_bbf \
--face-type big_big_face \
--jpeg-quality 100 \
--image-size 512 \
--max-faces-from-image 0


python main.py extract --detector s3fd \
--input-dir /home/ubuntu/data/psp/frame/Trump \
--output-dir /home/ubuntu/data/psp/frame/Trump_aligned_h \
--face-type head \
--jpeg-quality 100 \
--image-size 512 \
--max-faces-from-image 0





```