#!/usr/bin/env bash

# test on Kinetics-400 classes:
# ---------------------------------
#1: Download youtube files: [Works]
python download.py /storage/gby/datasets/Kinetics/kinetics-400_val_small.csv /storage/gby/datasets/Kinetics/val_small/videos
#2 Convert from avi to jpg files using utils/video_jpg_kinetics.py [Works]
python utils/video_jpg_kinetics.py /storage/gby/datasets/Kinetics/val_small/videos /storage/gby/datasets/Kinetics/val_small/jpgs
#3: Generate n_frames files using utils/n_frames_kinetics.py: [Works]
python utils/n_frames_kinetics.py /storage/gby/datasets/Kinetics/val_small/jpgs
#4: Generate annotation file in json format similar to ActivityNet using utils/kinetics_json.py [Works]
python utils/kinetics_json.py /storage/gby/datasets/Kinetics/kinetics-400_val_small.csv /storage/gby/datasets/Kinetics/kinetics-400_val_small.csv /storage/gby/datasets/Kinetics/kinetics-400_val_small.csv /storage/gby/datasets/Kinetics/kinetics-400_val_small.json
#5: test model on the downloaded files:
# with resnext-101-kinetics.pth TODO [Currently an open issue in GitHub]
python main.py --root_path /storage/gby/datasets/Kinetics/ --video_path val_small/jpgs --annotation_path kinetics-400_val_small.json --result_path results --dataset kinetics \
--n_classes 400 --model resnext --model_depth 101 --resnet_shortcut B --batch_size 2 --n_threads 4 --test_subset val --pretrain_path /storage/gby/models/C3Ds/resnext-101-kinetics.pth --no_train --no_val --test

# test on HMDB51: [Works]
# ---------------------------------
NOTE: YOU NEED TO USE PYTORCH < 4: therefore, downgrade pytorch:
conda install pytorch=0.3.0 cuda80 -c soumith

# test HDMB51 [Works]
python main.py --root_path /storage/gby/datasets/HMDB51 --video_path /storage/gby/datasets/HMDB51/jpgs --annotation_path /storage/gby/datasets/HMDB51/hmdb51_1.json \
--result_path results/test1 --dataset hmdb51 --n_classes 51 --n_finetune_classes 51 --test_subset val --resume_path /storage/gby/models/C3Ds/resnext-101-kinetics-hmdb51_split1.pth \
--test --no_train --no_val --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --batch_size 2 --n_threads 2

python main.py --root_path /storage/gby/datasets/HMDB51 --video_path /storage/gby/datasets/HMDB51/jpgs --annotation_path /storage/gby/datasets/HMDB51/hmdb51_1.json \
--result_path results/test1 --dataset hmdb51 --n_classes 51 --n_finetune_classes 51 \
--model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 16 \
--resume_path /storage/gby/models/C3Ds/resnext-101-kinetics-hmdb51_split1.pth \
--batch_size 16 --n_threads 4 --no_train --no_val --test --test_subset val

# eval the val.json file you received in /results: [Works]
python2.7
>>from utils import eval_hmdb51
>>hmdb = eval_hmdb51.HMDBclassification('/storage/gby/datasets/HMDB51/hmdb51_1.json', '/storage/gby/datasets/HMDB51/results/test1/val.json',verbose=True, top_k=1)
>>hmdb.evaluate()
# Error@1: 0.360130718954
# Error@5: 0.0947712418301

# test on UCF101: [Works]
# ---------------------------------
# test a ft_model for UCF101 # [Works]
python main.py --root_path /storage/gby/datasets/UCF101 --video_path jpgs --annotation_path ucfTrainTestlist/ucf101_01.json \
--result_path results/test1 --dataset ucf101 --n_classes 101 \
--model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 16 \
--resume_path /storage/gby/models/C3Ds/resnext-101-kinetics-ucf101_split1.pth \
--batch_size 16 --n_threads 4 --no_train --no_val --test --test_subset val

python2.7
>>from utils import eval_ucf101
>>ucf = eval_ucf101.UCFclassification('/storage/gby/datasets/UCF101/ucfTrainTestlist/ucf101_01.json', '/storage/gby/datasets/UCF101/results/test1/val.json',verbose=True, top_k=1)
# Error@1: 0.0985989955062
# Error@5: 0.0111022997621
# Should get 87.9% accuracy by using resnext-101-kinetics-ucf101_split1.pth. Reported is 88.9%

# -------------------------------------
# fine-tune Kinetics on UCF101:
# -------------------------------------
# -------------------------------------
# fine-tune resnet-34-kinetics.pth on ucf101_01.json: [Works]
python main.py --root_path ~/data --video_path /storage/gby/datasets/UCF101/jpgs --annotation_path /storage/gby/datasets/UCF101/ucfTrainTestlist/ucf101_01.json \
--result_path /storage/gby/datasets/UCF101/results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path /storage/gby/models/C3Ds/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 1 --n_threads 2 --checkpoint 5

# -------------------------------------
# fine-tune Kinetics on HMDB51:
# -------------------------------------
# -------------------------------------
# ft on resnet-34-kinetics.pth
python main.py --root_path /storage/gby/datasets/HMDB51 --video_path /storage/gby/datasets/HMDB51/jpgs --annotation_path /storage/gby/datasets/HMDB51/hmdb51_1.json \
--result_path results2 --dataset hmdb51 --n_classes 400 --n_finetune_classes 51 \
--pretrain_path /storage/gby/models/C3Ds/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 16 --n_threads 4 --checkpoint 10 --n_epochs 200

# ft on resnext-101-kinetics.pth
python main.py --root_path /storage/gby/datasets/HMDB51 --video_path /storage/gby/datasets/HMDB51/jpgs --annotation_path /storage/gby/datasets/HMDB51/hmdb51_1.json \
--result_path results/ft1 --dataset hmdb51 --n_classes 400 --n_finetune_classes 51 \
--pretrain_path /storage/gby/models/C3Ds/resnext-101-kinetics.pth --ft_begin_index 4 \
--model resnext --model_depth 101 --resnet_shortcut B  --resnext_cardinality 32 \
--batch_size 16 --n_threads 4 --checkpoint 50 --n_epochs 200

# python -m tensorboard.main --logdir='/storage/gby/datasets/HMDB51/results/ft1' --port=6006

# for activitynet: [Works]
python2.7 get_classification_performance.py data/activity_net.v1-3.min.json data/activity_net.v1-3.min.json


# -------------------------------------
# # fine-tune Kinetics on SIDB: TODO
# -------------------------------------
# -------------------------------------
#1: Download SIDB youtube files from csv file (Kinetics style): [Works]
cd ~/code/ActivityNet/Crawler/Kinetics
python download.py ~/Interactions/data/SIDB/splits/sidb2_split1_train.csv ~/Interactions/data/SIDB/videos
python download.py ~/Interactions/data/SIDB/splits/sidb2_split1_validation.csv ~/Interactions/data/SIDB/videos
python download.py ~/Interactions/data/SIDB/splits/sidb2_split1_test.csv ~/Interactions/data/SIDB/videos
#2 Convert from avi to jpg files using utils/video_jpg_kinetics.py [Works]
cd ~/code/3D-ResNets-PyTorch/
python utils/video_jpg_kinetics.py ~/Interactions/data/SIDB/videos ~/Interactions/data/SIDB/jpgs
#3: Generate n_frames files using utils/n_frames_kinetics.py: [Works]
python ~/code/3D-ResNets-PyTorch/utils/n_frames_kinetics.py ~/Interactions/data/SIDB/jpgs
#4: Generate annotation file in json format similar to ActivityNet using utils/kinetics_json.py [Works]
python ~/code/3D-ResNets-PyTorch/utils/kinetics_json.py ~/Interactions/data/SIDB/splits/sidb2_split1_train.csv \
~/Interactions/data/SIDB/splits/sidb2_split1_validation.csv ~/Interactions/data/SIDB/splits/sidb2_split1_test.csv ~/Interactions/data/SIDB/sidb2_split1.json
# fine-tune Kinetics on SIDB: [Works]
python main.py --root_path /afs/csail.mit.edu/u/g/gby/Interactions/data/SIDB --video_path /afs/csail.mit.edu/u/g/gby/Interactions/data/SIDB/jpgs --annotation_path sidb2_split1.json \
--result_path results --dataset sidb --n_classes 400 --n_finetune_classes 2 \
--pretrain_path /storage/gby/models/C3Ds/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 8 --n_threads 4 --checkpoint 30 --n_epochs 200

# resume path: TODO
python main.py --root_path /afs/csail.mit.edu/u/g/gby/Interactions/data/SIDB --video_path /afs/csail.mit.edu/u/g/gby/Interactions/data/SIDB/jpgs --annotation_path sidb2_split1.json \
--result_path results --dataset sidb --n_classes 400 --n_finetune_classes 2 \
--resume_path /afs/csail.mit.edu/u/g/gby/Interactions/data/SIDB/results/save_30.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 8 --n_threads 4 --checkpoint 30 --n_epochs 200

# TODO: EXPLORE
# I think 2000 videos dataset can be used for fine-tuning. To avoid overfitting, you should fine-tune a part of models, such as only last fc layer.
#--ft_begin_index option control how much layer is fine-tuned. In our experiments on UCF-101 and HMDB-51, fine-tuning only conv5_x and fc achieved the best results.

