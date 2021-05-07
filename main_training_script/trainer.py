import torch
import torch.nn as nn
import os
import shutil
import time

from utils.logger import Logger, savefig
from utils.evaluator import accuracy
from utils.misc import AverageMeter
from tqdm import tqdm

class Trainer(object):

    def __init__(self, model, args, data_loader, name):
        self.best_acc = 0
        
        self.args = args
        self.trainloader = data_loader[0]
        self.testloader = data_loader[1]
        self.model = model
        self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.start_epoch = args.start_epoch
        self.end_epoch = args.epochs

        title = name
        if args.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
            args.checkpoint = os.path.dirname(args.resume)
            self._resume(args.resume, title)
        else:
            self.logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            self.logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        
        if args.evaluate:
            print('\nEvaluation only')
            test_loss, test_acc = trainer.test(self.testloader, self.model, self.criterion, self.start_epoch)
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.adjust_learning_rate(epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.args.epochs, self.args.lr))
            train_loss, train_acc = self._train_one_epoch(epoch)
            test_loss, test_acc = self.test(epoch)

            self.logger.append([self.args.lr, train_loss, test_loss, train_acc, test_acc])
            
            # Saving Best Model
            is_best = test_acc > self.best_acc
            self.best_acc = max(test_acc, self.best_acc)
            self._save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'acc': test_acc,
                    'best_acc': self.best_acc,
                    'optimizer' : self.optimizer.state_dict(),
                }, is_best, checkpoint=self.args.checkpoint)

        self.logger.close()
        self.logger.plot()
        savefig(os.path.join(self.args.checkpoint, 'log.eps'))

        print('Best acc:')
        print(self.best_acc)

    def _train_one_epoch(self, epoch):
        self.model.train()
        self._reset_metric()
        end = time.time()

        tbar = tqdm(self.trainloader, ncols=130)
        
        for batch_idx, (inputs, targets) in enumerate(tbar):
            self.data_time.update(time.time() - end)

            # Main Output
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Metric
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            self.losses.update(loss.item(), inputs.size(0))
            self.top1.update(prec1.item(), inputs.size(0))
            self.top5.update(prec5.item(), inputs.size(0))

            # Gradient and Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Additional Things
            self.batch_time.update(time.time() - end)
            end = time.time()

            tbar.set_description('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
                        data=self.data_time.avg,
                        bt=self.batch_time.avg,
                        loss=self.losses.avg,
                        top1=self.top1.avg,
                        top5=self.top5.avg,
                    ))

        return (self.losses.avg, self.top1.avg)

    def test(self, epoch):
        self._reset_metric()
        self.model.eval()
        
        end = time.time()
        tbar = tqdm(self.testloader, ncols=130)
        for batch_idx, (inputs, targets) in enumerate(tbar):
            self.data_time.update(time.time() - end)

            # Main Output
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Metric
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            self.losses.update(loss.item(), inputs.size(0))
            self.top1.update(prec1.item(), inputs.size(0))
            self.top5.update(prec5.item(), inputs.size(0))

            # Additional Things
            self.batch_time.update(time.time() - end)
            end = time.time()

            tbar.set_description('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(self.testloader),
                    data=self.data_time.avg,
                    bt=self.batch_time.avg,
                    loss=self.losses.avg,
                    top1=self.top1.avg,
                    top5=self.top5.avg,
            ))
        
        return (self.losses.avg, self.top1.avg)

    def _resume(self, path, title):
        checkpoint = torch.load(path)
        self.best_acc = checkpoint['best_acc']
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger = Logger(os.path.join(self.args.checkpoint, 'log.txt'), title=title, resume=True)

    def _reset_metric(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def _save_checkpoint(self, state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.args.lr *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.lr
