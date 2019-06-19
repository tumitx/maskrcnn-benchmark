export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--config-file "configs/shufa_cfgs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml" \