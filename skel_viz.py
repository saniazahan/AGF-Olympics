#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:58:22 2022

@author: 22905553
"""
import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from numpy import random
import torch
import math

# AGF Gender Classes
actions = {
    1: "Female",
    2: "Male"
    }

## Openpose extraction
olympic_bone_pairs = tuple((i, j) for (i,j) in (
    (23,22),(22,11),(11,24),(11,10),(10,9),(9,8),(8,12),(12,13),(13,14),(14,21),(14,19),(19,20),
    (8,1),(1,0),(0,16),(16,18),(0,15),(15,17),(1,5),(5,6),(6,7),(1,2),(2,3),(3,4)
))

bone_pairs = {
    # NTU general
    'olympic': olympic_bone_pairs,
}

def visualize(skel):
    dataset = 'olympic'
    bones = bone_pairs[dataset]
    
    def animate(skeleton):
        ax.clear()
       
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        
        #ax.set_xlim([-2020,100])
        #ax.set_ylim([-1180,100])
        
        for i, j in bones:
            joint_locs = skeleton[:,[i,j]]            
            ax.plot(joint_locs[0],joint_locs[1], color='black', marker='o', markerfacecolor='tomato', linewidth=2.5, markersize=10)

        #plt.title('Skeleton {} Frame #{} of 3500)'.format(index, skeleton_index[0]))
        wait = input('Enter')
        #skeleton_index[0] += 1
        return ax
    
    skeleton_frames = skel
    
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ani = FuncAnimation(fig, animate, skeleton_frames, repeat = False)
                
    plt.title('Skeleton from {} test data'.format(dataset))
    plt.show()
    ani.save('heatmap_skel.gif', dpi=80, writer='imagemagick')
    