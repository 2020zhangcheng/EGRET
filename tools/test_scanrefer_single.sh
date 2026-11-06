TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port ${MASTER_PORT:-1111} \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ./data/ \
    --val_freq 3 --batch_size 16 --save_freq 3 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/ \
    --lr_decay_epochs 50 75 \
    --self_attend --augment_det \
    --model PVGround \
    --eval \
    --checkpoint_path ./data/PV-Ground_ScanRefer_single.pth \


# Evaluate a LoRA-finetuned checkpoint (appended args win, so the checkpoint and
# the model switches are overridden):
#   CUDA_VISIBLE_DEVICES=0,1,2 NGPU=3 \
#   EXTRA_ARGS="--checkpoint_path ./logs/scanrefer/<exp>/<ts>/ckpt_epoch_10.pth \
#               --lora --roi --max_epoch 10 --data_root ./data/ --batch_size 8" \
#   bash tools/test_scanrefer_single.sh
tools/__pycache__