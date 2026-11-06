#!/usr/bin/env bash
# Full evaluation of a LoRA-finetuned PVGround (ROI fusion enabled).
#
#   CKPT=./logs/PVGround_ScanRefer_LoRA_ROI/ckpt_epoch_10.pth \
#       bash tools/test_scanrefer_single_lora.sh
#
# The LoRA checkpoint only stores the adapters; the frozen pretrained weights are
# restored from --init_checkpoint automatically.
set -x
export PATH=/home/c_zhang/miniconda3/envs/3dgrounding/bin:$PATH
export PYTHONPATH=$PWD:$PWD/pointnet2:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1,2
NGPU=3


python -m torch.distributed.launch --nproc_per_node=${NGPU} --use_env --master_port=29512 \
    train_dist_mod.py \
    --num_decoder_layers 6 --num_target 256 \
    --use_color \
    --print_freq 50 --size_delta 0.1 --center_delta 6.0 --epoch_delta 1.0 \
    --batch_size 8 --num_workers 8 \
    --data_root ./data/ \
    --dataset scanrefer --test_dataset scanrefer --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align --self_attend \
    --model PVGround --log_dir ./logs/ --exp ${EXP:-PVGround_ScanRefer_LoRA_ROI_eval} \
    --eval --checkpoint_path logs/scanrefer/PVGround/1793872504/ckpt_epoch_6.pth \
    --init_checkpoint ./data/PV-Ground_ScanRefer_single.pth \
    --lora --lora_rank 8 --lora_alpha 16 \
    --roi --roi_points 512 --roi_topk 32 --roi_radius 0.3 
