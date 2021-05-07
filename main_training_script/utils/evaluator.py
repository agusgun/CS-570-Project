import torch
import numpy as np
from torchvision.utils import make_grid
import datetime
from utils.utils import Check_dir
import os
from torch.utils import tensorboard

value_scale = 255
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Exp_Recorder(object):

    def __init__(self,configs,name):
        self.configs=configs
        self.exp_time=datetime.datetime.now().strftime("%m_%d_%H_%M")
        self.saver_folder=os.path.join("run",name,self.exp_time)
        Check_dir(self.saver_folder)
        self.writer=tensorboard.SummaryWriter(self.saver_folder)
        self.best_pred=0.0


    def _save_model(self,state,epoch,class_Iou):
        save_name=os.path.join(self.saver_folder,'model_'+str(epoch)+'.pth')
        if(epoch>=1):
            delete_name=self.saver_folder+'/model_'+str(epoch-1)+'.pth'
            if(os.path.exists(delete_name)):
                os.remove(delete_name)
        torch.save(state,save_name)
        pre=state['best_prediction']
        if self.best_pred<pre:
            self.best_pred=pre
            best_name=os.path.join(self.saver_folder,'best_model.pth')
            torch.save(state,best_name)


    def _update_writer(self,loss,acc,optimizer,step,mode='train'):
        self.writer.add_scalar(''+str(mode)+'/total_loss',loss,step)
        self.writer.add_scalar(''+str(mode)+'/Cla_acc',acc.item(),step)

        if mode=='train':
            for i , para_group in enumerate(optimizer.param_groups):
                self.writer.add_scalar('lr',para_group['lr'],step)






class AverageMeter(object):
    "help computing and storing the metrics"
    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,value,count=1):
        self.val=value
        self.sum+=value*count
        self.count+=count
        self.avg=self.sum/self.count

    @property
    def value(self):
        return self.val

    def average(self):
        return np.round(self.avg,5)