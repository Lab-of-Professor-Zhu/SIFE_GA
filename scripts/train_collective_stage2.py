import sys
sys.path.append(".")
from train_net import *

cfg=Config('collective')

cfg.device_list="0"
cfg.training_stage=2
cfg.stage1_model_path='/home/computer/zkj/GA/Group-Activity-Recognition-master/result/[Collective_stage1_stage1]<2021-06-01_14-40-03>/stage1_epoch24_88.84%.pth'  #PATH OF THE BASE MODEL
cfg.train_backbone=False

cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13
cfg.num_actions=6
cfg.num_activities=5
cfg.num_frames=10
cfg.num_graph=4
cfg.tau_sqrt=True

cfg.batch_size=17
cfg.test_batch_size=1
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=100

cfg.exp_note='Collective_stage2'
train_net(cfg)