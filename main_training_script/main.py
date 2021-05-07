
import torch
import argparse
import os
from trainer import Trainer
from Model.model import Net_builter
import torchvision
import torchvision.transforms as transforms
from Model.sync_batchnorm import convert_model,DataParallelWithCallback

def main(args):
    model=Net_builter(backbone=args.backbone)

    if args.use_Sync_bn:
        model=convert_model(model)
        device=[i for i in range(len(args.device))]
        model=DataParallelWithCallback(model,device_ids=device)



    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    trainer=Trainer(model,args,[trainloader,testloader])

    trainer.training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-b', '--backbone', default='ResNet50', type=str,
                        help='the backbone we want to train on')
    parser.add_argument('-d', '--device', default='0', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-norl','--normalization',default='batch',help='choose the normalization method')

    parser.add_argument('--lr','--learning rate',default=0.01,type=float,help='choose the optimizer we want to use')

    parser.add_argument('--epoch',default=10,type=int)
    parser.add_argument('--resume',default=None,type=str)
    parser.add_argument('--eval',default=5,type=int,help='the interval of evaluation')
    parser.add_argument('--use_Sync_bn',default=True,type=bool,help='use sync_bn when having multiple GPU')


    args = parser.parse_args()
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)

