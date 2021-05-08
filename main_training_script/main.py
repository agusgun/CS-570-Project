
import torch
import argparse
import os
import torchvision
import torchvision.transforms as transforms

from trainer import Trainer
from models import build_network
from dataset import build_data_loader
from utils.misc import mkdir_p

def main(args):
    args.checkpoint = os.path.join('run', args.checkpoint)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    trainloader, testloader = build_data_loader(args.dataset, batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    num_classes = 0
    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imagenet-mini':
        num_classes = 1000

    model = build_network(args.net, num_classes, args.normalization)

    trainer = Trainer(model, args, [trainloader, testloader], name=args.checkpoint)
    if args.evaluate:
        pass
    else:
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    
    # Resume The Training
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    # Less Important Parameter
    parser.add_argument('-d', '--device', default='0', type=str,
                        help='indices of GPUs to enable (default: 0)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')

    # Important Params for Experiment
    parser.add_argument('-net', '--net', type=str,
                        help='the network we want to train on', choices=['ResNet50', 'VGG16'])
    parser.add_argument('-dataset', '--dataset', type=str, 
                        help='the dataset we want to use to train the network', 
                        choices=['imagenet-mini', 'cifar10', 'cifar100', 'svhn'])
    parser.add_argument('-norm', '--normalization', help='choose the normalization layer', 
                        choices=['bn', 'gn', 'gn_plus_sequential_bn_first', 'gn_plus_sequential_gn_first', 'gn_plus_parallel'])
    parser.add_argument('--lr', '--learning-rate', type=float, help='choose the learning rate')
    parser.add_argument('--batch_size', type=int, help='training batch size', choices=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--schedule', type=int, nargs='+',
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    args = parser.parse_args()
    print(args)
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)
