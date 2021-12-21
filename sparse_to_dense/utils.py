import os
import torch
import shutil
import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.viridis

def parse_command():
    #Set the depth groups as a tuple here if using --variable-scale
    # scaleMeans = (0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5)
    # scaleVariances = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

    scaleMeans = (0.8,1.0,1.3)
    scaleVariances = (0.0,0.0,0.0)

    model_names = ['resnet18', 'resnet50']
    loss_names = ['l1', 'l2']
    data_names = ['nyudepthv2', 'kitti', 'tof', 'tum']
    tof_names = ['flowerpower', 'trcnarrow', 'trcstandard', 'trcwide']
    from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo, StaticSampling, ProjectiveSampling, NearestSampling, ORBSampling
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo, StaticSampling, ProjectiveSampling, NearestSampling, ORBSampling]]
    from models import Decoder
    decoder_names = Decoder.names
    from dataloaders.dataloader import MyDataloader
    modality_names = MyDataloader.modality_names

    import argparse
    parser = argparse.ArgumentParser(description='Sparse-to-Dense')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                        help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv2', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('-b', '--batch-size', default=20, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--scale-min', default=1.0, type=float, metavar='scaleMin',
                        help='random image scaling minimum bound')
    parser.add_argument('--scale-max', default=1.5, type=float, metavar='scaleMax',
                        help='random image scaling maximum bound')
    parser.add_argument('--pixx', default=114, type=int, metavar='pixx',
                        help='pixel x location to use with static sampler after resizing to half')
    parser.add_argument('--pixy', default=152, type=int, metavar='pixy',
                        help='pixel y location to use with static sampler after resizing to half')
    parser.add_argument('--variable-focal', dest='varFocus', action='store_true',
                        help='simulate variable focal length data')
    parser.set_defaults(varFocus=False)
    parser.add_argument('--variable-scale', dest='varScale', action='store_true',
                        help='simulate variable scale (depth group) data')
    parser.set_defaults(pretrained=False)
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--crossTrain', dest='crossTrain', type=str, default='',
                        help='train old model using current input parameters, put the model path here')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    parser.add_argument('--tofType', metavar='TOFTYPE', default='flowerpower',
                        choices=tof_names,
                        help='dataset type when using data=tof: ' + ' | '.join(tof_names) + ' (default: flowerpower)')
    parser.add_argument('-r', '--ros', action='store_true',
                        default=False, help='Start module as ROS node.')
    parser.add_argument('--evaluate_tum', dest='evaluate_tum', type=str, default='',
                        help='evaluate model on the TUM datasets')
    parser.add_argument('--orb-noise', dest='orb_noise', action='store_true',
                        help='Simulate orb sampling noise')

    parser.set_defaults(pretrained=True)
    args, unknown = parser.parse_known_args()
    if args.modality == 'rgb' and args.num_samples != 0:
        # print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    if args.modality == 'rgb' and args.max_depth != 0.0:
        # print("max depth is forced to be 0.0 when input modality is rgb/rgbd")
        args.max_depth = 0.0
    if args.varScale:
        # print("\n\n         =========\nIMPORTANT: Variable depth groups are enabled, make sure to set these values at the top of the utils.py file\n         =========\n\n")
        if(isinstance(scaleMeans, tuple)):
            assert(len(scaleMeans) == len(scaleVariances))
    args.scaleMeans = scaleMeans
    args.scaleVariances = scaleVariances
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.2 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_output_directory(args):
    if(args.data == 'tof'):
        output_directory = os.path.join('results',
            '{}.{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}.varFocus={}.varScale={}.pixx={}.pixy={}'.
            format(args.data, args.tofType, args.sparsifier, args.num_samples, args.modality, \
                args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
                args.pretrained, args.varFocus, args.varScale, args.pixx, args.pixy))
    elif(args.sparsifier == 'statsam'):    
        output_directory = os.path.join('results',
            '{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}.varFocus={}.varScale={}.pixx={}.pixy={}'.
            format(args.data, args.sparsifier, args.num_samples, args.modality, \
                args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
                args.pretrained, args.varFocus, args.varScale, args.pixx, args.pixy))
    elif(args.sparsifier == 'orb'):
        output_directory = os.path.join('results',
        '{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}.crosstrained={}.varFocus={}.varScale={}.orbNoise={}'.
        format(args.data, args.sparsifier, args.num_samples, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained, args.crossTrain, args.varFocus, args.varScale, args.orb_noise))
    else:
        output_directory = os.path.join('results',
        '{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}.crosstrained={}.varFocus={}.varScale={}'.
        format(args.data, args.sparsifier, args.num_samples, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained, args.crossTrain, args.varFocus, args.varScale))
    return output_directory


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    
    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
