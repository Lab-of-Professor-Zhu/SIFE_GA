import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
import math

class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()
        
        self.cfg=cfg
        
        NFR =cfg.num_features_relation
        
        NG=cfg.num_graph
        N=cfg.num_boxes
        T=cfg.num_frames
        
        NFG=cfg.num_features_gcn
        NFG_ONE=NFG
        
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        
        if cfg.dataset_name=='volleyball':
            self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([T*N,NFG_ONE]) for i in range(NG) ])
        else:
            self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([NFG_ONE]) for i in range(NG) ])
        
            

        
    def forward(self,graph_boxes_features,boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """
        
        # GCN graph modeling
        # Prepare boxes similarity relation
        B,N,NFG=graph_boxes_features.shape
        NFR=self.cfg.num_features_relation
        NG=self.cfg.num_graph
        NFG_ONE=NFG
        
        OH, OW=self.cfg.out_size
        pos_threshold=self.cfg.pos_threshold
        
        # Prepare position mask
        graph_boxes_positions=boxes_in_flat  #B*T*N, 4
        graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2 
        graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2 
        graph_boxes_positions=graph_boxes_positions[:,:2].reshape(B,N,2)  #B*T, N, 2
        
        graph_boxes_distances=calc_pairwise_distance_3d(graph_boxes_positions,graph_boxes_positions)  #B, N, N
        
        position_mask=( graph_boxes_distances > (pos_threshold*OW) )
        
        
        relation_graph=None
        graph_boxes_features_list=[]
        for i in range(NG):
            graph_boxes_features_theta=self.fc_rn_theta_list[i](graph_boxes_features)  #B,N,NFR
            graph_boxes_features_phi=self.fc_rn_phi_list[i](graph_boxes_features)  #B,N,NFR

#             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
#             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph=torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N

            similarity_relation_graph=similarity_relation_graph/np.sqrt(NFR)

            similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
            
        
        
            # Build relation graph
            relation_graph=similarity_relation_graph

            relation_graph = relation_graph.reshape(B,N,N)

            relation_graph[position_mask]=-float('inf')

            relation_graph = torch.softmax(relation_graph,dim=2)       
        
            # Graph convolution
            one_graph_boxes_features=self.fc_gcn_list[i]( torch.matmul(relation_graph,graph_boxes_features) )  #B, N, NFG_ONE
            one_graph_boxes_features=self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features=F.relu(one_graph_boxes_features)
            
            graph_boxes_features_list.append(one_graph_boxes_features)
        
        graph_boxes_features=torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B, N, NFG
        
        return graph_boxes_features,relation_graph

class GCNnet_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(GCNnet_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph
        
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained=True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained=False)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.nl_emb_1=nn.LayerNorm([NFB])
        
        
        self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])    
        
        
        self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        self.fc_actions=nn.Linear(NFG,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFG,self.cfg.num_activities)

        self.gcn_RAM = nn.Linear(1024, 1024, bias=False)
        self.gcn_RAM_LN = nn.LayerNorm((1024))
        self.integration_param = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.integration = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.x = 0
        self.conv = nn.Linear(in_features=2, out_features=1024, bias=False)
        self.A = nn.Parameter(torch.FloatTensor([[0.0, 1.0, 1.0],[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]), requires_grad=False)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
    
    def scaledTanh(self, param):
        result = (torch.tanh(param)+1.0)/2.0
        return result
      
    def scaledTanh_(self, param):
        result = (torch.tanh(param))/2.0
        return abs(torch.tensor(result))
    
    def normalizeRAM(self, RAM):
      RAM = torch.where(RAM < 0.5, torch.zeros_like(RAM).cuda(), RAM)
      da = torch.sum(RAM, dim=0)  # sum over column
      db = torch.sum(RAM, dim=1)  # sum over line
      norm = torch.einsum('v,w->vw', (db, da))
      norm = torch.pow((norm+1e-5), -0.5)
      RAM = RAM*norm
      return RAM  
    
    def findMax(self, result):
        max_id_1 = 0;
        max_id_2 = 0;
        for i in range(len(result)):
            if result[i] < result[max_id_2]:
                if result[i] < result[max_id_1]:
                    max_id_1 = i;
                else:
                    max_id_2 = i;
        return max_id_1, max_id_2

    def KNN(self, boxes_in):
        knn_result = []
        pos_center = []
        for i in range(12):
            x = boxes_in[i, 0]
            y = boxes_in[i, 1]
            w = boxes_in[i, 2]
            h = boxes_in[i, 3]
            center_x = x + w/2
            center_y = y + h/2
            center = [center_x, center_y]
            pos_center.append(center)

        for i in range(12):
            result = []
            for j in range(12):
                if i==j :
                    result.append(10000)
                else:
                    distance = (pos_center[i][0] - pos_center[j][0]) * (pos_center[i][0] - pos_center[j][0]) + (pos_center[i][1] - pos_center[j][1]) *(pos_center[i][1] - pos_center[j][1])
                    distance = distance ** 0.5
                    result.append(distance)
            id1, id2 = self.findMax(result)
            knn_result.append([id1, id2])

        return knn_result

    def RAM_process(self, knn_result, boxes_features, boxes_in):
        # 输入这一帧下的KNN结果和特�?        
        for i in range(1):
            for j in range(len(knn_result)):
                if i==j :
                    continue
                else:
                    f_0 = torch.reshape(boxes_features[i,:], (1024, 1))
                    f_1 = torch.reshape(boxes_features[knn_result[i][0],:], (1024,1))
                    f_2 = torch.reshape(boxes_features[knn_result[i][1], :], (1024, 1))
                    f_2_0 = torch.reshape(boxes_features[j,:], (1024, 1))
                    f_2_1 = torch.reshape(boxes_features[knn_result[j][0], :], (1024, 1))
                    f_2_2 = torch.reshape(boxes_features[knn_result[j][1], :], (1024, 1))
                    group_1 = torch.cat((f_0, f_1), 1)
                    group_1 = torch.cat((group_1, f_2), 1)
                    group_2 = torch.cat((f_2_0, f_2_1), 1)
                    group_2 = torch.cat((group_2, f_2_2), 1)
                    
                    RAM_r = torch.einsum('tv,tw->vw', (group_1, group_2))
                    RAM_r = self.scaledTanh(RAM_r)
                    
                    group_1_pos = torch.cat((torch.reshape(boxes_in[i, :2],(1,2)), torch.reshape(boxes_in[knn_result[i][0], :2],(1,2))), 0)
                    group_1_pos = torch.cat((group_1_pos, torch.reshape(boxes_in[knn_result[i][1], :2],(1,2))), 0)
                    group_2_pos = torch.cat((torch.reshape(boxes_in[j, :2],(1,2)), torch.reshape(boxes_in[knn_result[j][0], :2],(1,2))), 0)
                    group_2_pos = torch.cat((group_2_pos, torch.reshape(boxes_in[knn_result[j][1], :2],(1,2))), 0)
                    
                    group_1_pos = self.conv(group_1_pos)
                    group_1_pos = torch.einsum('vt,wz->vw',(group_1_pos, self.A))
                    group_1_pos = torch.relu(group_1_pos)
                    
                    group_2_pos = self.conv(group_2_pos)
                    group_2_pos = torch.einsum('vt,wz->vw',(group_2_pos, self.A))
                    group_2_pos = torch.relu(group_2_pos)
                    
                    RAM_g_1 = torch.exp(group_1_pos)
                    RAM_g_2 = torch.exp(group_2_pos)
                    RAM_g = torch.einsum('vt,wt->vwt', (group_1_pos,group_2_pos))
                    RAM_g = (torch.log(RAM_g+1e-5)).pow(2)
                    RAM_g = (-1/2)*RAM_g
                    RAM_g_l = RAM_g[:,:,0]
                    RAM_g_r = RAM_g[:,:,1]
                    RAM_g = RAM_g_l + RAM_g_r
                    RAM_g = torch.exp(RAM_g)
                    
                    RAM_gen_percentile = self.scaledTanh_(self.integration_param)
                    RAM = RAM_gen_percentile * RAM_g + (1-RAM_gen_percentile) * RAM_r
                    
                    group_1 = group_1.t()
                    group_2 = group_2.t()
                    r_f_1 = self.gcn_RAM(torch.matmul(RAM, group_1))
                    r_f_1 = self.gcn_RAM_LN(r_f_1)
                    r_f_1 = F.relu(r_f_1)
                    r_f_2 = self.gcn_RAM(torch.matmul(RAM, group_2))
                    r_f_2 = self.gcn_RAM_LN(r_f_2)
                    r_f_2 = F.relu(r_f_2)
                    r_f_1 = r_f_1.t()
                    r_f_2 = r_f_2.t()
                    self.x = self.scaledTanh_(self.integration)
                    boxes_features[i, :] = boxes_features[i, :] + r_f_2[:, 0] * self.x
                    boxes_features[knn_result[i][0], :] = boxes_features[knn_result[i][0], :] + r_f_2[:, 1] * self.x
                    boxes_features[knn_result[i][1], :] = boxes_features[knn_result[i][1], :] + r_f_2[:, 2] * self.x
                    boxes_features[j, :] = boxes_features[j, :] + r_f_1[:, 0] * self.x
                    boxes_features[knn_result[j][0], :] = boxes_features[knn_result[j][0], :] + r_f_1[:, 1] * self.x
                    boxes_features[knn_result[j][1], :] = boxes_features[knn_result[j][1], :] + r_f_1[:, 2] * self.x
        return boxes_features    
                
    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        
        if not self.training:
            B=B*3
            T=T//3
            images_in.reshape( (B,T)+images_in.shape[2:] )
            boxes_in.reshape(  (B,T)+boxes_in.shape[2:] )
        
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
            
        
        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K
        
        
        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features)
        
        boxes_features_copy = boxes_features
             
        for batch_size in range(B):
            boxes_in = boxes_in.reshape(B,T,12,4)
            knn_result = self.KNN(boxes_in[batch_size, 0, :, :])
            for frame in range(T):  
                boxes_features[batch_size, frame, :, :] = self.RAM_process(knn_result, boxes_features[batch_size, frame, :, :], boxes_in[batch_size, frame, :, :]) 
        
        graph_boxes_feature=boxes_features_copy.reshape(B,T*N,NFG)
        
        for i in range(len(self.gcn_list)):
            graph_boxes_feature,relation_graph=self.gcn_list[i](graph_boxes_feature,boxes_in_flat)

        graph_boxes_feature=graph_boxes_feature.reshape(B,T,N,NFG)  
        boxes_features = boxes_features.reshape(B,T,N,NFB)
        
        boxes_states=boxes_features + graph_boxes_feature * (1-self.x)
        boxes_states=self.dropout_global(boxes_states)
        
        NFS=NFG
        
        # Predict actions
        boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        # Predict activities
        boxes_states_pooled,_=torch.max(boxes_states,dim=2)  
        boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFS)  
        
        activities_scores=self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        # Temporal fusion
        actions_scores=actions_scores.reshape(B,T,N,-1)
        actions_scores=torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores=activities_scores.reshape(B,T,-1)
        activities_scores=torch.mean(activities_scores,dim=1).reshape(B,-1)
        
        if not self.training:
            B=B//3
            actions_scores=torch.mean(actions_scores.reshape(B,3,N,-1),dim=1).reshape(B*N,-1)
            activities_scores=torch.mean(activities_scores.reshape(B,3,-1),dim=1).reshape(B,-1)
       
       
        return actions_scores, activities_scores
       
        

        
class GCNnet_collective(nn.Module):
    """
    main module of GCN for the collective dataset
    """
    def __init__(self, cfg):
        super(GCNnet_collective, self).__init__()
        self.cfg=cfg
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        
        self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        
        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.nl_emb_1=nn.LayerNorm([NFB])
        
        
        self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])    
        
        
        self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        self.fc_actions=nn.Linear(NFG,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFG,self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

#         nn.init.zeros_(self.fc_gcn_3.weight)
        

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
        
                
    def forward(self,batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data
        
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        MAX_N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        if not self.training:
            B=B*3
            T=T//3
            images_in.reshape( (B,T)+images_in.shape[2:] )
            boxes_in.reshape(  (B,T)+boxes_in.shape[2:] )
            bboxes_num_in.reshape((B,T))
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in=boxes_in.reshape(B*T,MAX_N,4)
                
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
            
        
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        

        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))  #B*T*MAX_N, 4
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))  #B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features_all=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*MAX_N, D, K, K,
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K
        
        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=self.nl_emb_1(boxes_features_all)
        boxes_features_all=F.relu(boxes_features_all)
        
        
        boxes_features_all=boxes_features_all.reshape(B,T,MAX_N,NFB)
        boxes_in=boxes_in.reshape(B,T,MAX_N,4)
        
    
        actions_scores=[]
        activities_scores=[]
        bboxes_num_in=bboxes_num_in.reshape(B,T)  #B,T,
        
        for b in range(B):
            
            N=bboxes_num_in[b][0]
            
            boxes_features=boxes_features_all[b,:,:N,:].reshape(1,T*N,NFB)  #1,T,N,NFB
        
            boxes_positions=boxes_in[b,:,:N,:].reshape(T*N,4)  #T*N, 4
        
            # GCN graph modeling
            for i in range(len(self.gcn_list)):
                graph_boxes_features,relation_graph=self.gcn_list[i](boxes_features,boxes_positions)
        
        
            # cat graph_boxes_features with boxes_features
            boxes_features=boxes_features.reshape(1,T*N,NFB)
            boxes_states=graph_boxes_features+boxes_features  #1, T*N, NFG
            boxes_states=self.dropout_global(boxes_states)
            

            NFS=NFG
        
            boxes_states=boxes_states.reshape(T,N,NFS)
        
            # Predict actions
            actn_score=self.fc_actions(boxes_states)  #T,N, actn_num
            

            # Predict activities
            boxes_states_pooled,_=torch.max(boxes_states,dim=1)  #T, NFS
            acty_score=self.fc_activities(boxes_states_pooled)  #T, acty_num
            
            
            # GSN fusion
            actn_score=torch.mean(actn_score,dim=0).reshape(N,-1)  #N, actn_num
            acty_score=torch.mean(acty_score,dim=0).reshape(1,-1)  #1, acty_num
            
            
            actions_scores.append(actn_score)  
            activities_scores.append(acty_score)
            
            

        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        activities_scores=torch.cat(activities_scores,dim=0)   #B,acty_num
        
        
        if not self.training:
            B=B//3
            actions_scores=torch.mean(actions_scores.reshape(-1,3,actions_scores.shape[1]),dim=1)
            activities_scores=torch.mean(activities_scores.reshape(B,3,-1),dim=1).reshape(B,-1)
       
        
#         print(actions_scores.shape)
#         print(activities_scores.shape)
       
        return actions_scores, activities_scores
        