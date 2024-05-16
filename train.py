import argparse
import collections
import os
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision import transforms

# from tensorboardX import SummaryWriter
import datetime

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

# print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='coco')
    parser.add_argument('--coco_path', help='Path to COCO directory', default=r"E:\Yangwenhui\Data\FJH2022\KCrossValidation\new10cross\group5")
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=250)

    ###
    parser.add_argument('--log', default='logs', help='save log')
    parser.add_argument('--results', default='results', help='save result')
    parser.add_argument('--times', default=5, help='save pth for xx epoch')
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--pretrained', default=True, help='True or False')

    parser.add_argument('--fn', default='BiFPN', help='FPN or BiFPN')  ##

    parser = parser.parse_args(args)

    # build dirs
    if not os.path.exists(parser.log):
        os.makedirs(parser.log)
    if not os.path.exists(parser.results):
        os.makedirs(parser.results)

    time = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')  # 时间
    os.makedirs(os.path.join(parser.log, f'{time}/weights'))
    # loss实时观测
    # writer = SummaryWriter(log_dir=f'{parser.log}/{str(time)}')

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))  # 学习如何处理数据集
        dataset_val = CocoDataset(parser.coco_path, set_name='val',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)  # batch_size设置
    dataloader_train = DataLoader(dataset_train, num_workers=parser.num_workers, collate_fn=collater,
                                  batch_sampler=sampler)  # num_workers设置

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained, FN=parser.fn)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained, FN=parser.fn)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)  # 学习率设置

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        reg_loss_last = None
        class_loss_last = None
        run_loss_last = None

        with tqdm(dataloader_train, mininterval=0.3) as q_bar:
            for iter_num, data in enumerate(q_bar):
                q_bar.set_description(f'{epoch_num}/{parser.epochs}')
                # try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                # 设置进度条显示
                q_bar.set_postfix({'Classification loss': '{:1.5f}'.format(float(classification_loss)),
                                   'Regression loss': '{:1.5f}'.format(float(regression_loss)),
                                   'Running loss': '{:1.5f}'.format(np.mean(loss_hist))})

                # print(
                #     'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running '
                #     'loss: {:1.5f}'.format(
                #         epoch_num, iter_num, float(classification_loss), float(regression_loss),
                #         np.mean(loss_hist)))
                if iter_num == len(q_bar) - 1:
                    reg_loss_last = '{:1.5f}'.format(float(regression_loss))
                    class_loss_last = '{:1.5f}'.format(float(classification_loss))
                    run_loss_last = '{:1.5f}'.format(np.mean(loss_hist))

                del classification_loss
                del regression_loss
                # except Exception as e:
                #     print(e)
                #     continue

            if parser.dataset == 'coco':
                if (epoch_num + 1) % parser.times == 0:
                    print('Evaluating dataset')
                    coco_eval.evaluate_coco(dataset_val, retinanet, type='train', save_epoch=epoch_num + 1)

            elif parser.dataset == 'csv' and parser.csv_val is not None:

                print('Evaluating dataset')

                mAP = csv_eval.evaluate(dataset_val, retinanet)

            scheduler.step(np.mean(epoch_loss))  ###

        with open(os.path.join(parser.log, f'{time}/log.csv'), 'a+') as loss_file:
            # print('保存loss值：', class_loss_last, reg_loss_last, run_loss_last)
            loss_file.write(
                f"{str(epoch_num)},'Classification_loss',{class_loss_last}\n")
            loss_file.write(
                f"{str(epoch_num)},'Regression_loss',{reg_loss_last}\n")
            loss_file.write(
                f"{str(epoch_num)},'Running_loss',{run_loss_last}\n")
            # writer.add_scalar('Classification_loss/train', '{:1.5f}'.format(class_loss_last), epoch_num + 1)
            # writer.add_scalar('Regression_loss/train', '{:1.5f}'.format(reg_loss_last), epoch_num + 1)
            # writer.add_scalar('Running loss/train', '{:1.5f}'.format(run_loss_last), epoch_num + 1)

        if (epoch_num + 1) % parser.times == 0:
            # torch.save(retinanet.module, '{}/{}/weights/retinanet_{}.pt'.format(parser.log, time, epoch_num + 1))
            torch.save(retinanet, '{}/{}/weights/retinanet_{}.pt'.format(parser.log, time, epoch_num + 1))

    # writer.close()

    retinanet.eval()

    torch.save(retinanet, '{}/{}/weights/model_final.pt'.format(parser.log, time))


if __name__ == '__main__':
    main()
