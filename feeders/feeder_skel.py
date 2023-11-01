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


class Feeder(Dataset):
    def __init__(self, data_path, label_path, video_path=None,
                 random_choose=False, random_shift=False, random_move=False, 
                 window_size=-1, normalization=False, augmentation=False, debug=False, bone=False, use_mmap=True, crop=False, p_interval=1):
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
        shear_amplitude = 0.5
        temperal_padding_ratio = 6
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.crop = crop
        self.p_interval = p_interval
        
        dataset = 'ntu'
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
        if normalization:
            self.get_mean_map()
        
        
            
    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label, self.sample_class = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label, self.sample_class = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if 'train' in self.data_path:
            T = 1000
        else:
            T = 100   
            
        if self.debug:
            self.label = self.label[0:T]
            self.data = self.data[0:T]
            self.sample_name = self.sample_name[0:T]
    
    def vid_transform(self, img):
        #mean = [0.45, 0.45, 0.45]
        #std = [0.225, 0.225, 0.225]
        #frames_per_second = 25
        transforms = T.Compose([
                    T.Resize((300,300)),
                    T.ToTensor(),
                    T.Normalize(
                            mean=[0.45, 0.45, 0.45],
                            std=[0.225, 0.225, 0.225]
                        )
                ])
        img = transforms(img)
        return img
    
    def get_video_data(self, index):
        max_frames = 3567
        sport = {'0':'floor','1':'ring'}
        sample_name = self.sample_name[index]
        #print(sample_name)
        if 'train' in self.label_path:
            split = 'train'
        else:
            split = 'test'
        clss = sport[sample_name.split('_')[-2]]
        path = self.video_path+split+'/'+clss+'/'+sample_name+'/'
        #print(path)
        files = os.listdir(path)
        #print(len(files))
        files.sort()
        #print(files[0])
        fr_idx = np.arange(1,len(files)-1)
        other_fs = np.random.choice(fr_idx, 23, replace=False)
        
        frames = np.zeros((56,3, 300, 300), dtype=np.float32)
        with Image.open(path+files[0]) as im:
            img = self.vid_transform(im)
        frames[0] = img
        with Image.open(path+files[-1]) as im:
            img = self.vid_transform(im)
        frames[-1] = img
        k = 1
        for fr in enumerate(other_fs):
            #img = self.vid_transform(cv2.imread(path+files[0]))
            with Image.open(path+files[fr[1]]) as im:
                img = self.vid_transform(im)
            frames[k] = img
            k += 1
            #break
        
        return frames,56
    
    def get_video_frames(self, index):
        max_frames = 1500
        sport = {'0':'floor','1':'ring'}
        sample_name = self.sample_name[index]
        #print(sample_name)
        if 'train' in self.label_path:
            split = 'train'
        else:
            split = 'test'
        clss = sport[sample_name.split('_')[-2]]
        path = self.video_path+split+'/'+clss+'/'+sample_name+'/'+sample_name+'.npy'
        frames = np.load(path, mmap_mode='r')
        
        return frames, 0#frames, i
    
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
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        clss = self.sample_class[index]

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
                if np.isnan(label):
                    label = 14.3
                if self.video_path:
                    video_frame, seq_len = self.get_video_data(index)
                    return data_numpy, video_frame, seq_len, label, index
                else:
                    return data_numpy, label, clss, index

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


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
