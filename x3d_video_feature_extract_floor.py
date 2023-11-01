#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:52:44 2022

@author: 22905553

video models https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html

"""
#  cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
import torch
import torch.nn as nn
from torchvision import models
from feeders.feeder import Feeder
import numpy as np
import datetime
import pickle
#torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
class VideoModel(nn.Module):
            def __init__(self):
                super(VideoModel, self).__init__()
                model_name = 'x3d_l'#'mvit_base_32x3'   X3D_L - 77.44% and mvit_base_32x3 - 80.30 on kinetics400 
                #model_name = 'mvit_base_32x3'#'mvit_base_32x3'   X3D_L - 77.44% and mvit_base_32x3 - 80.30 on kinetics400 
                original_model =  torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
                self.features = nn.Sequential(
                    # stop at conv4
                    #*list(original_model.children())[:] #mvit
                    *list(original_model.blocks.children())[:-1] #x3d
                )
                
            def forward(self, x):
                x = self.features(x)
                return x
            
if __name__ == "__main__":
# cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    import thop
    from thop import clever_format
      
    
    model = VideoModel().cuda()
    model.eval()
    
    # x = torch.randn(1, 3, 16, 224,224)#.cuda()
    # macs, params = thop.profile(model, inputs=(x,), verbose=False)
    # macs, params = clever_format([macs, params], "%.2f")
    # print( macs, params)    
    
    splits=["all"] #['test', 'train']
    out_path = "/home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Data/X3D_extracted_video_features/"
               
    data_path = None #"/media/22905553/F020DDF820DDC5AE/Olympic_v1/Olympic_Openpose/train_joint_tracked_interpolated.npy"
    video_path = "/mnt/F020DDF820DDC5AE/AQA_Olympics/Olympic_v1/video_frames/"   
    data_loader = dict()
    
    for split in splits:
        #label_path = "/home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Data/Olympic_Openpose/Olympic_Openpose_V1/all_label_1st_index.pkl"
        label_path = "/mnt/F020DDF820DDC5AE/AQA_Olympics/Olympic_v1/Olympic_Openpose/old_data/test_label_v1.pkl"
        train_dataset = Feeder(data_path, label_path, video_path, seq_len=-1, video_only = True, modal = 'video_x3d', label_smooth=True, random_choose=False, random_shift=False, random_move=False, window_size=-1, normalization=False, augmentation=False, debug=False, bone=False, use_mmap=True, crop=False, p_interval=1)
        
        data_loader[split] = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False)
    
    
        print(split)
        for data, l, clss, name in data_loader[split]:
            #break
            sample_names = []
            #break
            d = data.permute(0,2,1,3,4)
            n = data.shape[1]
            print(name[0],': ', n)
            segment = 0        
            while segment < n: 
                inputs = d[:,:,segment:segment+16,:,:].float().cuda()
                start = datetime.datetime.now()
                preds = model(inputs)
                #print(preds.shape)
                #break
                if segment == 0:
                    temp = preds.detach().cpu()#.numpy()
                else:
                    temp = torch.cat((temp, preds.detach().cpu()), dim=2)
                segment += 16                
                inputs = inputs.detach().cpu()
                del inputs
                del preds
                torch.cuda.empty_cache()
                #print(temp.shape)
            end = datetime.datetime.now()
            delta = end - start
            print(int(delta.total_seconds() * 1000))
            temp = temp.cpu().numpy()
            #print(temp.shape)
            #new_data.append(temp)
            sample_names.append(name[0])
            #break
    
            #with open(f'{out_path}{name[0]}_video_features.pkl', 'wb') as f:
            #    pickle.dump((sample_names, temp), f)
            del temp
            del sample_names

    


    
    
