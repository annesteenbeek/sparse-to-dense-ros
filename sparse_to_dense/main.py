#!/usr/bin/env python

import os
import time
import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from models import ResNet
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo, StaticSampling, ProjectiveSampling, NearestSampling, ORBSampling
import criteria
import utils

args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'margin10', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']

tum_rooms = [
    'rgbd_dataset_freiburg1_room',
    'rgbd_dataset_freiburg3_long_office_household',
    'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
    'rgbd_dataset_freiburg3_structure_texture_far'
]

best_result = Result()
best_result.set_to_worst()

def write_csv(filename, avg):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writerow(vars(avg))


def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join('data', args.data, 'train')
    valdir = os.path.join('data', args.data, 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == StaticSampling.name:
        sparsifier = StaticSampling(pixx=args.pixx, pixy=args.pixy)
    elif args.sparsifier == ProjectiveSampling.name:
        sparsifier = ProjectiveSampling()
    elif args.sparsifier == NearestSampling.name:
        sparsifier = NearestSampling(pixx=args.pixx, pixy=args.pixy)
    elif args.sparsifier == ORBSampling.name:
        sparsifier = ORBSampling(num_samples=args.num_samples, max_depth=max_depth, add_noise=args.orb_noise)
    else:
        print("Unknown sparsifier")

    if args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier, augArgs=args)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier, augArgs=args)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier, augArgs=args)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier, augArgs=args)

    elif args.data == 'tof':
        traindir = os.path.join('data', args.tofType, 'train')
        valdir = os.path.join('data', args.tofType, 'val')
        from dataloaders.tof_dataloader import TOFDataset
        if not args.evaluate:
            train_dataset = TOFDataset(traindir, type='train',
                modality=args.modality, sparsifier=StaticSampling(), augArgs=args)
        val_dataset = TOFDataset(valdir, type='val',
            modality=args.modality, sparsifier=StaticSampling(), augArgs=args)
    elif args.data == 'tum':
        if not args.evaluate:
            raise RuntimeError('TUM dataset only used for evaluation')
        from dataloaders.tum_dataloader import TUMDataset
        valdir = os.path.join('data', 'tum', args.tum_room)
        val_dataset = TUMDataset(valdir, type='val', 
            modality=args.modality, sparsifier=sparsifier, augArgs=args)
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti or tof or tum.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def main():
    global args, best_result, output_directory, train_csv, test_csv, fieldnames

    # evaluation mode
    start_epoch = 0
    if args.ros:
        # import ros objects later to avoid ros dependencies
        import rospy
        from ros.rosNode import ROSNode 

        rospy.init_node('sparse_to_dense')
        best_model_filename = rospy.get_param("~model_path")

        assert os.path.isfile(best_model_filename), \
        "=> no best model found at '{}'".format(best_model_filename)
        print("=> loading best model '{}'".format(best_model_filename))
        checkpoint = torch.load(best_model_filename)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))

        rosnode = ROSNode(model)
        rosnode.run()
        return
 
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        args.evaluate = True
        _, val_loader = create_data_loaders(args)
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return
    if args.evaluate_tum:
        assert os.path.isfile(args.evaluate_tum), \
        "=> no best model found at '{}'".format(args.evaluate_tum)
        print("=> loading best model '{}'".format(args.evaluate_tum))
        checkpoint = torch.load(args.evaluate_tum)
        output_directory = os.path.dirname(args.evaluate_tum)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        args.data = "tum"
        args.evaluate = True

        # TODO also validate NYU?

        tum_csv = os.path.join('results', 'results_overview.csv')
        fieldnames.insert(0, "dataset")
        fieldnames.insert(0, "directory")
        if not os.path.exists(tum_csv):
            with open(tum_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # also add best result
        best_result.dataset = "best"
        best_result.directory = output_directory
        write_csv(tum_csv, best_result)

        args.sparsifier = "orb"

        for room in tum_rooms:
            print("Evaluating dataset: %s" % room)
            args.tum_room = room
            _, val_loader = create_data_loaders(args)
            avg, _ = validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
            avg.dataset = room + "_ORB"
            avg.directory = output_directory
            write_csv(tum_csv, avg)
        return

    
    elif args.crossTrain:
        print("Retraining loaded model on current input parameters")
        train_loader, val_loader = create_data_loaders(args)
        checkpoint = torch.load(args.crossTrain)
        model = checkpoint['model']
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)
        model = model.cuda()

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True

    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.arch == 'resnet50':
            model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)

        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model = model.cuda()
        # from torchsummary import summary
        # summary(model, input_size=(in_channels, 228, 304))

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
        result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        }, is_best, epoch, output_directory)


def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    with tqdm(total=len(train_loader)) as t:
        t.set_description('Epoch %d' % epoch)
        for i, (input, target) in enumerate(train_loader):

            input, target = input.cuda(), target.cuda()
            torch.cuda.synchronize()
            data_time = time.time() - end

            # compute pred
            end = time.time()
            pred = model(input)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward() # compute gradient and do SGD step
            optimizer.step()
            torch.cuda.synchronize()
            gpu_time = time.time() - end

            # measure accuracy and record loss
            result = Result()
            result.evaluate(pred.data, target.data)
            average_meter.update(result, gpu_time, data_time, input.size(0))
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                avg = average_meter.average()
                t.set_postfix(gpu_time=gpu_time, RMSE=avg.rmse, MAE=avg.mae)
            t.update()

    avg = average_meter.average()
    write_csv(train_csv, avg)

def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    with tqdm(total=len(val_loader)) as t:
        t.set_description("validating")
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            torch.cuda.synchronize()
            data_time = time.time() - end

            # compute output
            end = time.time()
            with torch.no_grad():
                pred = model(input)
            torch.cuda.synchronize()
            gpu_time = time.time() - end

            # measure accuracy and record loss
            result = Result()
            result.evaluate(pred.data, target.data)
            average_meter.update(result, gpu_time, data_time, input.size(0))
            end = time.time()

            # save 8 images for visualization
            skip = 50
            if args.modality == 'd':
                img_merge = None
            else:
                if args.modality == 'rgb':
                    rgb = input
                elif args.modality == 'rgbd':
                    rgb = input[:,:3,:,:]
                    depth = input[:,3:,:,:]

                if i == 0:
                    if args.modality == 'rgbd':
                        img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                    else:
                        img_merge = utils.merge_into_row(rgb, target, pred)
                elif (i < 8*skip) and (i % skip == 0):
                    if args.modality == 'rgbd':
                        row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                    else:
                        row = utils.merge_into_row(rgb, target, pred)
                    img_merge = utils.add_row(img_merge, row)
                elif i == 8*skip:
                    filename = output_directory + '/comparison_' + str(epoch) + '.png'
                    utils.save_image(img_merge, filename)

            if (i+1) % args.print_freq == 0:
                avg = average_meter.average()
                t.set_postfix(gpu_time=gpu_time, RMSE=avg.rmse, MAE=avg.mae)
            t.update()

    avg = average_meter.average()

    print('RMSE={average.rmse:.3f} '\
        'MAE={average.mae:.3f} '\
        'Delta1={average.delta1:.3f} '\
        'REL={average.absrel:.3f} '\
        'Lg10={average.lg10:.3f} '\
        'Margin10={average.margin10:.3f} '\
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        write_csv(test_csv, avg)

    return avg, img_merge

if __name__ == '__main__':
    main()
