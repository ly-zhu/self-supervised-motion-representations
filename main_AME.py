# System libs
import os
import random
import time

# Torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F

# Numerical and imaging libs
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
from PIL import Image
import pylab

from mir_eval.separation import bss_eval_sources

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder
from utils import AverageMeter, recover_rgb, warpgrid, makedirs, cmap_im, plot_onlyFinalerr_metrics


def normalize_sfs(sfs, scale = 255.):
    scale = torch.tensor(scale, dtype=sfs.dtype)
    return torch.sign(sfs)*(torch.log1p(1 + scale*torch.abs(sfs)) / torch.log1p(1 + scale))


# Network wrapper, defines forward pass
class NetWrapper1(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper1, self).__init__()
        self.net_sound = nets

    def forward(self, audios, audios_shift, args):
        B = audios[0].size()[0]
        N = args.num_mix

        preds = [None for n in range(N)]
        preds_shift = [None for n in range(N)]
        for n in range(N):
            # audio normalization
            audios[n] = normalize_sfs(audios[n])
            audios_shift[n] = normalize_sfs(audios_shift[n])
            preds[n], preds_shift[n] = self.net_sound(audios[n], audios_shift[n])
        return preds, preds_shift


# Network wrapper, defines forward pass
class NetWrapper2(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper2, self).__init__()
        self.net_frame = nets

    def forward(self, frames, args):
        B = frames[0].size()[0]        
        N = args.num_mix

        cams = [None for n in range(N)]
        cam = [None for n in range(N)]
        preds = [None for n in range(N)]
        for n in range(N):
            _, cams[n], preds[n], _ = self.net_frame.forward_att(frames[n])
            cam[n] = torch.abs(cams[n])
            cam[n] = F.adaptive_avg_pool3d(cam[n], (1,None, None))
            cam[n] = torch.squeeze(cam[n], 2)
        return cam, preds


def cam_fun(frames, cam, lo_frac = 0.5, adapt = True, max_val = 35):
    """ Set heatmap threshold adaptively, to deal with large variation in possible input videos. """
    # im: B, 3, 224, 224
    # cam 224, 224

    max_prob = 0.35
    if adapt:
      max_val = np.percentile(cam, 97)

    same = np.max(cam) - np.min(cam) <= 0.001
    if same:
      return frames
    
    outs = []
    for i in range(frames.shape[0]):
        lo = lo_frac * max_val
        hi = max_val + 0.001
        im = frames[i]
        f = cam.shape[0] * float(i) / frames.shape[0]
        l = int(f)
        r = min(1 + l, cam.shape[0]-1)
        p = f - l
        frame_cam = ((1-p) * cam[l]) + (p * cam[r])
        vis = cmap_im(pylab.cm.jet, frame_cam, lo = lo, hi = hi)
        p = np.clip((frame_cam - lo)/float(hi - lo), 0, max_prob)
        p = p[..., None]
        im = np.array(im, 'd')
        vis = np.array(vis, 'd')
        outs.append(np.uint8(im*(1-p) + vis*p))
    return np.array(outs)


def save(img_fname, a):
    if img_fname.endswith('jpg'):
        return Image.fromarray(np.uint8(a)).save(img_fname, quality = 100)
    else:
        return Image.fromarray(np.uint8(a)).save(img_fname, quality = 100)


# Visualize predictions
def output_visuals_overlay(vis_rows, batch_data, masks, args):
    # fetch data and predictions
    frames = batch_data['frames']
    infos = batch_data['infos']
    audios = batch_data['audios']
    audios_shift = batch_data['audios_shift']

    # unwarp log scale
    N = args.num_mix
    B = frames[0].size(0)
    masks = masks.detach().cpu().numpy()
    audios[0] = audios[0].detach().cpu().numpy()
    audios_shift[0] = audios_shift[0].detach().cpu().numpy()

    threshold = 0.5
    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []

        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        frames_tensor = recover_rgb(frames[0][j,:,int(args.num_frames//2)])
        frames_tensor = np.asarray(frames_tensor)
        filename_frame = os.path.join(prefix, 'frame.png')
        matplotlib.image.imsave(os.path.join(args.vis, filename_frame), frames_tensor)

        frame = frames_tensor.copy()
        height, width = masks.shape[-2:]
        heatmap = np.zeros((height, width))
   
        vis = cam_fun(frame[np.newaxis], masks[j], adapt = True)
        path_overlay = os.path.join(args.vis, prefix, 'overlay.jpg')
        save(path_overlay, vis[0])
        
        vis_rows.append(row_elements)


def evaluate(crit, netWrapper1, netWrapper2, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=False)

    # switch to eval mode
    netWrapper1.eval()
    netWrapper2.eval()

    # initialize meters
    loss_meter = AverageMeter()

    vis_rows = []
    N = args.num_mix
    for i, batch_data in enumerate(loader):
        # forward pass
        frames = batch_data['frames']
        audios = batch_data['audios']
        audios_shift = batch_data['audios_shift']
        for n in range(N):
            frames[n] = torch.autograd.Variable(frames[n]).to(args.device)
            audios[n] = torch.autograd.Variable(audios[n]).to(args.device)
            audios_shift[n] = torch.autograd.Variable(audios_shift[n]).to(args.device)

        preds_auds, preds_auds_shift = netWrapper1.forward(audios, audios_shift, args)
        cam, preds_imgs = netWrapper2.forward(frames, args)
        mask = F.interpolate(cam[0], size=(frames[0].shape[-2], frames[0].shape[-1]), mode = 'bilinear')
 
        err = crit(preds_imgs[0], preds_auds[0], preds_auds_shift[0]).reshape(1)
        err = err.mean()

        # visualization preparation
        BS, WS = preds_auds[0].shape
        preds_imgs[0] = preds_imgs[0].view(BS, 1, 1, WS)
        preds_imgs_mask = F.interpolate(preds_imgs[0], size=(1, audios[0].shape[-1]), mode = 'bilinear')

    
        loss_meter.update(err.item())
        print('[Eval] iter {}, loss: {:.4f},  '.format(i, err.item()))

        # output visualization
        if len(vis_rows) < args.num_vis:
            output_visuals_overlay(vis_rows, batch_data, mask, args)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '.format(epoch, loss_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_onlyFinalerr_metrics(args.ckpt, history)


# train one epoch
def train(crit, netWrapper1, netWrapper2, loader, optimizer, history, epoch, args):
    print('Training at {} epochs...'.format(epoch))
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # set to train mode
    netWrapper1.train()
    netWrapper2.train()

    N = args.num_mix
    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        
        data_time.update(time.perf_counter() - tic)

        # forward pass
        optimizer.zero_grad()
        frames = batch_data['frames']
        audios = batch_data['audios']
        audios_shift = batch_data['audios_shift']
        for n in range(N):
            frames[n] = torch.autograd.Variable(frames[n]).to(args.device)
            audios[n] = torch.autograd.Variable(audios[n]).to(args.device)
            audios_shift[n] = torch.autograd.Variable(audios_shift[n]).to(args.device)

        preds_auds, preds_auds_shift = netWrapper1.forward(audios, audios_shift, args)
        cam, preds_imgs = netWrapper2.forward(frames, args)

        err = crit(preds_imgs[0], preds_auds[0], preds_auds_shift[0]).reshape(1)
        err = err.mean()
        
        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, '
                  'loss: {:.4f}, '
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, 
                          err.item())
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def checkpoint(nets, optimizer, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    state = {'epoch': epoch, \
             'state_dict_net_sound': net_sound.state_dict(), \
             'state_dict_net_frame': net_frame.state_dict(),\
             'optimizer': optimizer.state_dict(), \
             'history': history, }

    torch.save(state, '{}/checkpoint_{}'.format(args.ckpt, suffix_latest)

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(state, '{}/checkpoint_{}'.format(args.ckpt, suffix_best))

def load_checkpoint(nets, optimizer, history, filename):
    (net_sound, net_frame) = nets
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        net_sound.load_state_dict(checkpoint['state_dict_net_sound'])
        net_frame.load_state_dict(checkpoint['state_dict_net_frame'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        history = checkpoint['history']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    nets = (net_sound, net_frame)
    return nets, optimizer, start_epoch, history

def load_checkpoint_from_train(nets, filename):
    (net_sound, net_frame) = nets
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        print('epoch: ', checkpoint['epoch'])
        net_sound.load_state_dict(checkpoint['state_dict_net_sound'])
        net_frame.load_state_dict(checkpoint['state_dict_net_frame'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    nets = (net_sound, net_frame)
    return nets


def create_optimizer(nets, args):
    (net_sound, net_frame) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_synthesizer *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch='resnet182d_A',
        num_frames=args.num_frames,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch='resnet183d_V',
        weights=args.weights_frame)

    nets = (net_sound, net_frame)
    crit = nn.TripletMarginLoss(margin=2.0, p=2)

    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),#2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))


    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': []}}


    # Load model weights for evaluation and for continue training 
    # Current model is trained from scratch, you can change setting accordingly to initialize model parameters from exxisting pretrained networks
    start_epoch = 1
    model_name = args.ckpt + '/checkpoint_best.pth'
    if os.path.exists(model_name):
        if args.mode == 'eval':
            nets = load_checkpoint_from_train(nets, model_name)
        elif args.mode == 'train':
            model_name = args.ckpt + '/checkpoint_latest.pth'
            nets, optimizer, start_epoch, history = load_checkpoint(nets, optimizer, history, model_name)
            print("Loading from previous checkpoint.")
    

    # Wrap networks
    netWrapper1 = NetWrapper1(net_sound)
    netWrapper1 = torch.nn.DataParallel(netWrapper1, device_ids=range(args.num_gpus)).cuda()
    netWrapper1.to(args.device)
    netWrapper2 = NetWrapper2(net_frame)
    netWrapper2 = torch.nn.DataParallel(netWrapper2, device_ids=range(args.num_gpus)).cuda()
    netWrapper2.to(args.device)


    # Eval mode
    if args.mode == 'eval':
        evaluate(crit, netWrapper1, netWrapper2, loader_val, history, 0, args)
        print('Evaluation Done!')
        return

    # Train mode
    for epoch in range(start_epoch, args.num_epoch + 1):    
        train(crit, netWrapper1, netWrapper2, loader_train, optimizer, history, epoch, args)

        # adjust learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

        ## Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(crit, netWrapper1, netWrapper2, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(nets, optimizer, history, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    if args.mode == 'train':
        args.vis = os.path.join(args.ckpt, 'visualization_train/')
        makedirs(args.ckpt, remove=False)#True)
    elif args.mode == 'eval':
        args.vis = os.path.join(args.ckpt, 'visualization_val/')
    elif args.mode == 'test':
        args.vis = os.path.join(args.ckpt, 'visualization_test/')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
