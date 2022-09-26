import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

def conv2x2(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(15, 1),
        stride=stride,
        padding=(7, 0),
        bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

def downsample_aud_basic_block(x, planes, stride):
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Aud_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Aud_BasicBlock, self).__init__()
        self.conv1 = conv2x2(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2x2(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_V(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_channels = 3,
                 shortcut_type='B',
                 num_classes=21):#400):
        self.inplanes = 64
        super(ResNet_V, self).__init__()
        self.conv1 = nn.Conv3d(
            num_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=1)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=(1,2,2))
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1)
        self.fc = nn.Conv3d(
            512,
            1,
            kernel_size=1,
            stride=(1, 1, 1),
            bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, imgs):
        imgs= self.conv1(imgs)
        imgs = self.bn1(imgs)
        imgs = self.relu(imgs)
        imgs = self.maxpool(imgs) 

        imgs = self.layer1(imgs) 
        imgs = self.layer2(imgs) 
        imgs = self.layer3(imgs) 
        imgs = self.layer4(imgs) 
        
        imgs_C1 = self.fc(imgs) 
        imgs_CHW1 = torch.mean(imgs_C1, (3,4), keepdim=True)
        (B, CI, TI, HI, WI) = imgs.size()
        imgs_CHW1 = imgs_CHW1.view(B, TI) 
        imgs_THW1 = F.adaptive_avg_pool3d(imgs, 1)
        imgs_THW1 = imgs_THW1.view(B, CI)

        return imgs, imgs_C1, imgs_CHW1, imgs_THW1


class ResNet_A(nn.Module):
    def __init__(self, 
                aud_block,
                aud_layers,
                num_frames,
                shortcut_type='B'):
        self.aud_inplanes = 64
        super(ResNet_A, self).__init__()

        # audio branch
        self.aud_conv1 = nn.Conv2d(1, 64, kernel_size=(65, 1), stride = 4)
        self.aud_bn1 = nn.BatchNorm2d(64)
        self.aud_relu = nn.ReLU(inplace=True)
        self.aud_maxpool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4,1))
        self.aud_layer1 = self._make_aud_layer(aud_block, 128, aud_layers[0], shortcut_type, stride=(4,1))
        self.aud_layer2 = self._make_aud_layer(
            aud_block, 128, aud_layers[1], shortcut_type, stride=(4,1))
        self.aud_layer3 = self._make_aud_layer(
            aud_block, 256, aud_layers[2], shortcut_type, stride=(4,1))
        self.aud_frac_maxpool = nn.FractionalMaxPool2d(kernel_size=(4,1), output_size=(int(num_frames/4), 1))
        self.aud_fc = nn.Conv2d(256, 1, kernel_size=1)

    def _make_aud_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.aud_inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_aud_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.aud_inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.aud_inplanes, planes, stride, downsample))
        self.aud_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.aud_inplanes, planes))

        return nn.Sequential(*layers)


    def forward_once(self, auds):
        # audio branch
        (B, HS) = auds.size()
        auds = auds.view(B, 1, HS, 1)
        auds = self.aud_conv1(auds)
        auds = self.aud_bn1(auds)
        auds = self.aud_relu(auds)
        auds = self.aud_maxpool(auds)
        auds = self.aud_layer1(auds)
        auds = self.aud_layer2(auds)
        auds = self.aud_layer3(auds)
        auds = self.aud_frac_maxpool(auds)
        auds = self.aud_fc(auds) 

        (B, CS, TS, FS) = auds.size()
        auds = auds.view(B, TS) 
        return auds


    def forward(self, auds1, auds2):
        # forward pass of input 1
        auds1 = self.forward_once(auds1)
        # forward pass of input 2
        auds2 = self.forward_once(auds2)
        return auds1, auds2



def resnet18_V(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet_V(BasicBlock, [2, 2, 2, 2], 3, **kwargs)
    return model

def resnet18_A(num_frames = '48', **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet_A(Aud_BasicBlock, [2, 2, 2], num_frames, **kwargs)
    return model
