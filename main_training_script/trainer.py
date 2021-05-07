import torch
import numpy as np
from utils.utils import Poly
from tqdm import tqdm
import torch.nn as nn
from utils.evaluator import AverageMeter,Exp_Recorder

class Trainer(object):

    def __init__(self,model,args,Data_loader,name='CIFAR'):
        self.mode,self.val_step='train_',0
        self.model=model
        self.start_epoch=1
        self.end_epoch=args.epoch
        self.nbr_classes=10
        self.eval_interval=args.eval
        self.loss=nn.CrossEntropyLoss()
        self.train_loader=Data_loader[0]
        self.val_loader=Data_loader[1]
        self.loader_name=name
        self.saver=Exp_Recorder(configs=args,name=self.loader_name)



        if isinstance(self.model,torch.nn.DataParallel):
            trainable_parameters=[{'params': filter(lambda p:p.requires_grad, self.model.module.parameters())}]
        else:
            trainable_parameters=filter(lambda p:p.requires_grad,self.model.parameters())

        self.optimizer=torch.optim.SGD(trainable_parameters,args.lr)
        self.lr_scheduler=Poly(self.optimizer,self.end_epoch,len(self.train_loader))

        if args.resume:
            self._resume(path=args.resume)



    def training(self):
        self.model.train()
        for epoch in range(self.start_epoch,self.end_epoch+1):
            self._train_epoch(epoch)
            if (epoch%self.eval_interval):
                self._val_epoch(epoch)

    def _train_epoch(self,epoch):
        tbar=tqdm(self.train_loader,ncols=130)
        self._reset_metric()
        print("Current Learning Rate: {:.5f}          Current Best Prediction : {:.4f}".format(self.lr_scheduler.get_lr()[0],self.saver.best_pred))
        for step, (input,target) in enumerate(tbar):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            output=self.model(input)
            loss=self.loss(output,target)
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step(epoch=epoch-1)

            prediction=output.max(1)[1]
            acc=(prediction==target).float().sum()/target.shape[0]

            self._update_metrics(loss,acc)

            tbar.set_description('Train ({}) |cla_loss:{:.4f}|cla_acc:{:.4f}|'.format(
                epoch,self.total_loss.average(),
            self.total_acc.average()))
        self.saver._update_writer(self.total_loss.average(),self.total_acc.average(),self.optimizer,epoch,mode='train')


    def _val_epoch(self,epoch):
        tbar=tqdm(self.val_loader,ncols=130)
        self.model.eval()
        self._reset_metric()
        for step, (input,target) in enumerate(tbar):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output=self.model(input)
            loss=self.loss(output,target)
            prediction=output.max(1)[1]

            acc=(prediction==target).float().sum()/target.shape[0]

            self._update_metrics(loss,acc)

            tbar.set_description('Eval ({}) |cla_loss:{:.4f}|cla_acc:{:.4f}|'.format(
                epoch,self.total_loss.average(),
            self.total_acc.average()))

        self.saver._update_writer(self.total_loss.average(),self.total_acc.average(),self.optimizer,epoch,mode='Eval')
        self._save_checkpoint(epoch,self.total_acc.average())



    def _resume(self,path):

        checkpoint=torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.start_epoch=checkpoint['epoch']

    def _reset_metric(self):
        self.total_loss=AverageMeter()
        self.total_acc=AverageMeter()

    def _update_metrics(self,cla_loss,acc):
        self.total_loss.update(cla_loss.item())
        self.total_acc.update(acc.item())



    def _save_checkpoint(self,epoch,acc):
        state={
            'state_dict': self.model.state_dict(),
            'epoch':epoch,
            'optimizer':self.optimizer.state_dict(),
            'best_prediction':acc
        }
        self.saver._save_model(state,epoch,acc)