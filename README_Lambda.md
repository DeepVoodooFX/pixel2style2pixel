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
--exp_dir=outputs \
--checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=inputs/merge_TH_416_ref \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs
```
