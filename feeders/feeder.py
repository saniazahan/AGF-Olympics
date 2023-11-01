import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from feeders import tools
import random
import os
import cv2
import torchvision.transforms as T
from PIL import Image
#import cv2
import os
import feeders.videotransforms as videotransforms
from sklearn import preprocessing
import torch.nn.functional as F

class Feeder(Dataset):
    def __init__(self, data_path, label_path, video_path=None, seq_len=16, joint_feature=False, video_feature=False, video_only = False, modal = 'image',
                 label_smooth = True, random_choose=False, random_shift=False, random_move=False, 
                 window_size=-1, normalization=False, augmentation=False, debug=False, bone=False, 
                 use_mmap=True, crop=False, p_interval=1):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """
        
        self.debug = debug
        self.video_feature = video_feature
        self.modal = modal
        self.label_smooth = label_smooth
        self.seq_len = seq_len
        self.video_only = video_only
        self.augmentation = augmentation
        self.data_path = data_path
        self.video_path = video_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.bone = bone
        self.joint_feature = joint_feature
        shear_amplitude = 0.5
        temperal_padding_ratio = 6
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.crop = crop
        self.p_interval = p_interval
        
            
        dataset = 'ntu'
        if self.data_path:
            if 'kinetics' in self.data_path:
                self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            elif 'ntu' in self.data_path:
                self.connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
            elif 'ucla' in self.data_path:    
                self.connect_joint = np.array([2,2,2,3,3,5,6,7,3,9,10,11,1,13,14,15,1,17,18,19]) - 1
    
            self.case = 0
            if 'ntu' in self.data_path:
                if self.case == 0:
                    self.theta = 0.3
            elif self.case == 1:
                self.theta = 0.5
            elif 'ntu120' in self.data_path:
                self.theta = 0.3
            elif 'NW_UCLA' in self.data_path:
                self.theta = 0.17
        
        self.load_data()
        if normalization and self.data_path and self.joint_feature==False:
            self.get_mean_map()
        
        
    def label_smother(self, label):       
        #label = np.array(label).reshape(len(label),1)
        #min_max_scaler = preprocessing.MinMaxScaler()
        #label = min_max_scaler.fit_transform(label)
        #label = list(label[:,0])
        
        ## AGF-Olympics
        mn = 10.966
        mx = 16.2
        ## MIT-Skates
        #mn = 30.88
        #mx = 99.86
        label_scaled = list((np.array(label)-mn) / (mx-mn))

        return label_scaled
    
    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label, self.sample_class = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label, self.sample_class = pickle.load(f, encoding='latin1')

        if self.label_smooth:
            self.label = self.label_smother(self.label)
        
        # load data
        if self.data_path and self.joint_feature==False:
            if self.use_mmap:
                self.data = np.load(self.data_path, mmap_mode='r')
            else:
                self.data = np.load(self.data_path)
        if self.data_path and 'train' in self.data_path:
            T = 1000
        else:
            T = 100   
            
        if self.debug:
            self.label = self.label[0:T]
            if self.data_path:
                self.data = self.data[0:T]
            self.sample_name = self.sample_name[0:T]
    
   
    def cv_transform_image(self, img):
        '''
        transforms.Compose([
                    transforms.Resize((h,w)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)]) 
        '''
        if 'train' in self.label_path:
            if self.modal == 'image':
                transforms = T.Compose([#videotransforms.CenterCrop(224),
                                    T.Resize(size=(224,224)),
                                    videotransforms.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                                           ])
            elif self.modal == 'video_x3d':
                transforms = T.Compose([#videotransforms.CenterCrop(224),
                                    T.Resize(size=(256,256)),
                                    videotransforms.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize(
                                            mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
                                           ])
            elif self.modal == 'video_mvit':
                transforms = T.Compose([#videotransforms.CenterCrop(224),
                                    T.Resize(size=(224,224)),
                                    videotransforms.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize(
                                            mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
                                           ])
        else:
            if self.modal == 'image':
                transforms = T.Compose([T.Resize(size=(224,224)),
                                    #videotransforms.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])   
            elif self.modal == 'video_x3d':
                transforms = T.Compose([#videotransforms.CenterCrop(224),
                                    T.Resize(size=(256,256)),
                                    #videotransforms.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize(
                                            mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
                                           ])
            elif self.modal == 'video_mvit':
                transforms = T.Compose([#videotransforms.CenterCrop(224),
                                    T.Resize(size=(224,224)),
                                    #videotransforms.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize(
                                            mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
                                           ])
            
        img_transformed = transforms(img)
        #print(img_transformed.shape)
        return img_transformed
    
    def __pad_frames(self, frames, N):
        #feature = torch.from_numpy(feature).permute(0,2,3,1)
        T,C,H,W = frames.shape
        
        #print(feature.size(-1))
        #n = (6115 - T)#//2
        n = (N - T)#//2
        
        #p1d = (0, n)
        #feature = F.pad(feature, p1d, "constant", 0).permute(0,3,1,2).numpy()
        pad = np.zeros((C,n,H,W))
        pad = np.zeros((n,C,H,W))
        padded_frames = np.concatenate((frames,pad), axis=0)
        return padded_frames
    
    def load_rgb_frames_with_cv(self, index):
        max_frames = 6115
        sport = {'0':'floor','1':'ring'}
        sample_name = self.sample_name[index]
        #print(sample_name)
        if 'train' in self.label_path:
            split = 'train'
        else:
            split = 'test'
            
        #if "UNLV" in self.label_path and "diving" in self.label_path:
        #    clss = "diving"
        #else:
        #    clss = sport[sample_name.split('_')[-2]]
        #path = self.video_path+split+'/'+clss+'/'+sample_name#+'/'
        path = self.video_path+split+'/floor/'+sample_name#+'/'
        files = os.listdir(path)
        #print(len(files))
        #fr_idx = np.arange(1,len(files)-1)
        #fr_idx = [f.split('.')[0][5:] for f in files]
        
        fr_idx = [f.split('.')[0] for f in files]
        if "UNLV" in self.label_path and "diving" in self.label_path:
            other_fs = fr_idx
        elif "UNLV" in self.label_path and "skating" in self.label_path:
            if self.seq_len != -1:
                other_fs = np.random.choice(fr_idx, self.seq_len, replace=False)
            elif len(fr_idx)>5823  and self.seq_len == -1:
                other_fs = np.random.choice(fr_idx, 5823, replace=False)
            else:
                other_fs = fr_idx
        elif "MTL_AQA" in self.label_path:
            if self.seq_len != -1:
                other_fs = np.random.choice(fr_idx, self.seq_len, replace=False)
            elif len(fr_idx)>274  and self.seq_len == -1:
                other_fs = np.random.choice(fr_idx, 274, replace=False)
            else:
                other_fs = fr_idx
        else:
            if self.seq_len != -1:
                other_fs = np.random.choice(fr_idx, self.seq_len, replace=False)
            elif len(fr_idx)>2938  and self.seq_len == -1:
                other_fs = np.random.choice(fr_idx, 2938, replace=False)
            else:
                other_fs = fr_idx
        other_fs.sort()
        #print(other_fs)
        frames = []
        for i,fr in enumerate(other_fs):
            #print(os.path.join(path, 'frame'+ str(fr) + '.jpg'))
            #print(fr)
            if "MTL_AQA" in self.label_path:
                f_path = os.path.join(path, str(fr) + '.jpg')
            else:
                f_path = os.path.join(path, str(fr) + '.jpg')
            #print(f_path)
            #break
            try:
                img = cv2.imread(f_path)#[:, :, [2, 1, 0]]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                #if sample_name=='3_37':
                #    print(img.shape)
                w, h, c = img.shape
                if w < 224 or h < 224:
                    d = 224. - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
                #img = img.reshape(c,h,w)
                #img = (img / 255.)# * 2 - 1
                #print(img)
    
                #print(img.shape)
                img = Image.fromarray(img)
                #print(img.shape)
                img = self.cv_transform_image(img)
                #print(img.shape)
                #img = (img / 255.) #* 2 - 1
                frames.append(img)
            except:
                print(f_path)
                    
            #break
        if len(frames)==0:
            print(sample_name)
        #print(len(frames))
        frames = torch.stack(frames)
        if "UNLV" in self.label_path and "diving" in self.label_path:
            return frames
        if "UNLV" in self.label_path and "vault" in self.label_path:
            return frames
        elif "UNLV" in self.label_path and "skating" in self.label_path:
            if self.seq_len == -1 and len(frames)<5823:
                frames = self.__pad_frames(frames, 5823)
                #print(frames.shape)
            return frames
        elif "MTL_AQA" in self.label_path:
            if self.seq_len == -1 and len(frames)<274:
                frames = self.__pad_frames(frames, 274)
                #print(frames.shape)
            return frames
        else:
            if self.seq_len == -1 and len(frames)<2938:
                frames = self.__pad_frames(frames, 2938)
            return frames
            
            
    
    
    def __pad_feature(self, feature):
        #feature = torch.from_numpy(feature).permute(0,2,3,1)
        C,T,H,W = feature.shape
        
        #print(feature.size(-1))
        #n = (6115 - T)#//2
        n = (2938 - T)
        #p1d = (0, n)
        #feature = F.pad(feature, p1d, "constant", 0).permute(0,3,1,2).numpy()
        pad = np.zeros((C,n,H,W))
        padded_feature = np.concatenate((feature,pad), axis=1)
        return padded_feature
    
    def get_video_feature(self, index):
        sample_name = self.sample_name[index]
        #print(sample_name)
        
        #clss = sport[sample_name.split('_')[-2]]
        if 'test' in self.label_path:
            path = self.video_path+'/test/'+sample_name+'_video_features.pkl'
        else:
            path = self.video_path+'/train/'+sample_name+'_video_features.pkl'
        try:
            with open(path) as f:
                sample_names, feature = pickle.load(f)
        except:
            # for pickle file from python2
            with open(path, 'rb') as f:
                sample_names, feature = pickle.load(f, encoding='latin1')
        feature = feature[0]
        #if feature.shape[1]<2938:#
        if feature.shape[1]<2938 and "Olympic_v1" in self.label_path:
            feature = self.__pad_feature(feature)
        return feature
    
    def get_joint_feature(self, index):
        sample_name = self.sample_name[index]
        #print(sample_name)
        
        #clss = sport[sample_name.split('_')[-2]]
        path = self.data_path+'/'+sample_name+'_joint_features.pkl'
        try:
            with open(path) as f:
                sample_names, feature = pickle.load(f)
        except:
            # for pickle file from python2
            with open(path, 'rb') as f:
                sample_names, feature = pickle.load(f, encoding='latin1')
        #print(feature[0].shape)
        #print(len(feature))
        feature = feature[0]        
        
        return feature
    
    def get_mean_map(self):
        data = self.data
        #print(data.shape)
        #N, C, T, V, M = data.shape
        N, C, T, V = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 1, 3)).reshape((N * T, C * V)).std(axis=0).reshape((C, 1, V))
        self.std_map[self.std_map==0.0] = 0.0001
        
        
    def __sampler__(self):
        class_sample_count = np.array([len(np.where(self.label==t)[0]) for t in np.unique(self.label)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.label])

        samples_weight = torch.from_numpy(samples_weight)
        self.sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return  self.sampler
    
    def _aug(self, data_numpy):
        
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        axis, angle, std = random.randint(0, 2), random.randint(10, 90) , random.uniform(1, 9) * 0.001
        #if self.shear_amplitude > 0:
        data_numpy = tools._augmented_pair(data_numpy, axis, angle, std)
        
        return data_numpy
    
    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = self.label[index]
        
        clss = self.sample_class[index]
        name = self.sample_name[index]
        #print(name)
        if self.video_only:
            if self.video_feature:
                vid_feat = self.get_video_feature(index)
                #print(vid_feat.shape)
                return vid_feat, label, clss, name
            else:
                video_frame = self.load_rgb_frames_with_cv(index)
                return video_frame, label, clss, name
        
        elif self.joint_feature and self.video_feature:
            vid_feat = self.get_video_feature(index)
            joint_feat = self.get_joint_feature(index)
            return [joint_feat,vid_feat], label, clss, name
        elif self.joint_feature:
            joint_feat = self.get_joint_feature(index)
            return joint_feat, label, clss, name
        else:
            data_numpy = self.data[index]
            data_numpy = np.array(data_numpy)
        

            if self.normalization:
                data_numpy = (data_numpy - self.mean_map) / self.std_map
            if self.random_shift:
                data_numpy = tools.random_shift(data_numpy)
            if self.random_choose:
                data_numpy = tools.random_choose(data_numpy, self.window_size)
            elif self.window_size > 0:
                data_numpy = tools.auto_pading(data_numpy, self.window_size)
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            if self.crop:
                valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
                # reshape Tx(MVC) to CTVM
                data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
            if self.augmentation:
                #print('yes')
                #data_numpy = tools._transform(data_numpy, self.theta)
                # processing
                data1 = data_numpy#self._aug(data_numpy)
                data2 = self._aug(data_numpy)    
            
                return [data1, data2], label, index
            else:
            
                if self.bone:
                    C, T, V, M = data_numpy.shape  
                    bone = np.zeros((C, T, V, M))
                    for i in range(len(self.connect_joint)):
                        bone[:C,:,i,:] = data_numpy[:,:,i,:] - data_numpy[:,:,self.connect_joint[i],:]
                
                    return [data_numpy, bone] , label, index
                    #return [joint, velocity], label, index
                else:
                    if self.video_feature:
                        vid_feat = self.get_video_feature(index)
                        return [data_numpy,vid_feat], label, clss, index
                    else:
                        return data_numpy, label, clss, name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
