import torch
import torchvision
from torch import nn
import torch.nn.init as init

from .audio_visual_net import resnet18_V, resnet18_A

import numpy as np
import math

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            y = m.in_features
            m.weight.data.normal_(0.0, 1/np.sqrt(y))


    # builder for sound network
    def build_sound(self, arch='resnet182d_A', num_frames=48, weights=''):
        # 2D models
        if arch == 'resnet182d_A':
            net_sound = resnet18_A(num_frames=num_frames)
            net_sound.apply(self.weights_init)
            return net_sound
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    
    # builder for vision network
    def build_frame(self, arch='resnet183d_V', weights=''):
        if arch == 'resnet183d_V':
            net = resnet18_V()
        else:
            raise Exception('Architecture undefined!')

        #net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net