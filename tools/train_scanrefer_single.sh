TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port ${MASTER_PORT:-4444} \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ./data/ \
    --val_freq 2 --batch_size 8 --save_freq 2 --print_freq 500 \
    --lr_backbone=1e-4 --lr=5e-5 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/ \
    --lr_decay_epochs 20 50 \
    --self_attend --augment_det \
    --max_epoch 20 \
    --model PVGround \
    --exp PVGround_all_ft_20ep \
    --frozen \
    --lr_scheduler cosine \
    --roi --roi_points 512 --roi_topk 32 --roi_radius 0.4 \
    --checkpoint_path ./data/PV-Ground_ScanRefer_single.pth \


# LoRA fine-tuning with entropy-gradient guided ROI fusion (appended args win,
# so they override the defaults above; effective batch = NGPU * batch_size * accum_steps):
#   CUDA_VISIBLE_DEVICES=0,1,2 NGPU=3 \
#   EXTRA_ARGS="--max_epoch 10 --val_freq 2 --save_freq 2 --batch_size 8 --accum_steps 1 \
#               --lr 1e-4 --lr-scheduler cosine --lora --roi \
#               --checkpoint_path ./data/PV-Ground_ScanRefer_single.pth \
#   bash tools/train_scanrefer_single.sh
# Resume from epoch 5 (LoRA-only checkpoint + frozen base):
#   EXTRA_ARGS="... --resume --checkpoint_path ./logs/scanrefer/<exp>/<ts>/ckpt_epoch_4.pth \
#               --init_checkpoint ./data/PV-Ground_ScanRefer_single.pth"
