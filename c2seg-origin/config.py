# -*- coding: utf-8 -*-
import os
import ml_collections

tensorboard = True
deterministic = True
visualization = False
resume = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 922

task_name = "MosMedData+"  # Options: "MosMedData+", "QaTa-COV19"
model_name = "demo"
cosineLR = True
epochs = 500
es_patience = 50
save_after = 50
batch_size = 32
img_size = 224
n_channels = 3
n_classes = 1
token_len = 18
lr = 3e-4
loss_weight = [0.5, 0.2, 0.15, 0.1, 0.05]

dataset_root  = os.path.join(system_path, "datasets", task_name)
save_root = os.path.join(system_path, "projects", model_name, task_name)

# ===============================
# ViT-specific configuration
# ===============================
def get_ViT_config():
    config = ml_collections.ConfigDict()
    config.base_channel = 64
    config.clip_backbone = "ViT-B/32"   # e.g. ViT-B/16, ViT-B/32
    config.dropout = True
    config.dropout_value = 0.5
    config.text_mask_rate = 0.3
    config.img_mask_rate = 0.3
    config.mask_mode = "dist"          # Options: 'dist', 'random'
    config.mask_mode_dist_random = True
    config.pool_mode = "max_pool"      # Options: 'max_pool', 'aver_pool'
    config.rec_trans_num_layers1 = 3   # Number of transformer layers for reconstruction
    config.frozen_clip = True
    config.transformer = ml_collections.ConfigDict()
    return config

