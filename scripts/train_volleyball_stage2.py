import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.device_list="0"
cfg.training_stage=2
cfg.stage1_model_path=r'/home/computer/GCJ/Group_activity_recognition/Group-Activity-Recognition-master/result/[Volleyball_stage1_stage1]<2021-04-28_12-54-40>/stage1_epoch12_87.66%.pth'  #PATH OF THE BASE MODEL
cfg.train_backbone=False

cfg.batch_size=3 #32
cfg.test_batch_size=1
cfg.num_frames=3
cfg.train_learning_rate=2e-4 
cfg.lr_plan={41:1e-4, 81:5e-5, 121:1e-5}
cfg.max_epoch=150
cfg.actions_weights=[[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]  

cfg.exp_note='Volleyball_stage2'
train_net(cfg)