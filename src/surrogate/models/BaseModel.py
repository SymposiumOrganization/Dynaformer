import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
import numpy as np
import pandas as pd


def preprocessing(batch):
    '''preprocessing function'''
    x=batch[:,:,0].unsqueeze(2)
    y=batch[:,:,1].unsqueeze(2)

    # Randomize initial value and length of the sequence
    max_init = 50
    length = 50
    x_init = np.random.randint(0, max_init)
    xx, yy = x[:,x_init:x_init+length,:], y[:,x_init:x_init+length,:]

    # n=50
    # xx,yy=x[:,:n,:],y[:,:n,:] # take first 50 samples -> goes to encoder
    return xx, yy, x, y
    #query_curr=torch.cat([query_curr,query_curr_temp.unsqueeze(0)])

class BaseModel2(LightningModule):
    """Base class for all models, contains common functions such as training_step, validation_step, etc."""
    def __init__(self):
        super().__init__()
        self.results = []
        self.counter = 0

    def training_step(self, input, batch_idx):
        '''needs to return a loss from a single batch'''
        try:
            padded_current, padded_voltage, padded_xx, padded_yy, padded_tt,_= input
        except:
            padded_current, padded_voltage, padded_xx, padded_yy, padded_tt= input
        #xx, yy, x, y = preprocessing(batch)
        inp = torch.cat([padded_xx.unsqueeze(2),padded_yy.unsqueeze(2),padded_tt.unsqueeze(2)],axis=2)
        out = self.forward(inp,padded_current)
        zero_mask = padded_voltage != 0 # zero only for padding

        # Create weighted loss
        factor = 0.01
        weights = torch.arange(0,(padded_voltage.shape[1]-0.5)*factor,factor,device=self.device).repeat(input[0].shape[0],1) + 1
        #
        #weights = torch.ones(weights.shape,device=self.device)

        rmse_loss = self.loss_func(out[zero_mask].squeeze(), padded_voltage[zero_mask].squeeze())
        weights = weights[zero_mask]
        out = out[zero_mask].squeeze()
        vol = padded_voltage[zero_mask].squeeze()
        weigheted_loss = self.weighted_mse_loss(out,vol,weights)
        
        #loss = self.loss_func(out[zero_mask].squeeze(), padded_voltage[zero_mask].squeeze())
        self.log('train_loss_rmse', rmse_loss,on_step=True,on_epoch=True)
        self.log('train_loss_rmse_weighted', weigheted_loss,on_step=True,on_epoch=True)

        if self.loss == "rmse":
            loss = rmse_loss
        elif self.loss == "weighted_loss":
            loss = weigheted_loss
        else:
            raise KeyError("Loss function not recognized")

        self.log('train_loss', loss,on_step=True,on_epoch=True)

        return loss

    def validation_step(self, input, batch_idx):
        '''used for logging metrics'''
        val_idx = 0 
        if val_idx == 0:
            padded_current, padded_voltage, padded_xx, padded_yy, padded_tt, metadata= input
        else:
            padded_current, padded_voltage, padded_xx, padded_yy, padded_tt= input
        #xx, yy, x, y = preprocessing(batch)
        inp = torch.cat([padded_xx.unsqueeze(2),padded_yy.unsqueeze(2),padded_tt.unsqueeze(2)],axis=2)
        # if True:
        #     padding = torch.zeros([inp.shape[0],100,inp.shape[2]], device=self.device)
        #     inp = torch.cat([inp,padding],axis=1)
        zero_mask = padded_voltage != 0 # zero only for padding
        if val_idx == 0:
            #gt_voltage = padded_voltage[padded_voltage!=0]
            pred_voltages = self.forward(inp, padded_current)
            #out1 = self.forward(inp[-1:,:,:],padded_current[-1:,:])
            batch_entries = []
            errors = []
            for idx, current_pred in enumerate(pred_voltages):
                entry = {}
                entry["dataset"] = metadata["dataset"][idx]
                entry["curve"] = metadata['curve'][idx]
                current_ratio = metadata['ratio'][idx]
                entry["ratio"] = current_ratio
                is_dead =  bool(current_pred[-1] < 3.2)
                entry["is_dead"] =is_dead
                batch_entries.append(entry)
                if is_dead and current_ratio < 1:
                    errors.append(1-current_ratio)
                elif not is_dead and current_ratio > 1:
                    errors.append(current_ratio-1)
                else:
                    errors.append(0)
            self.results.append(pd.DataFrame(batch_entries))
        else:
            out = self.forward(inp,padded_current)
            loss = self.loss_func(out[zero_mask].squeeze(), padded_voltage[zero_mask].squeeze())
            self.log('valid_loss',loss,on_step=True,on_epoch=True)


    def validation_epoch_end(self, outputs) -> None:
        df = pd.concat(self.results)
        error_smaller_than_1 = df.loc[(df['ratio'] < 1)]
        error_bigger_than_1 = df.loc[(df['ratio'] > 1)]

        final_entries = []
        for dataset_curve in error_smaller_than_1.groupby(['dataset','curve']):
            dataset, curve = dataset_curve[0]
            curr_df = dataset_curve[1]
            failed = curr_df.loc[curr_df['is_dead']]
            if len(failed) == 0:
                ratio = 1
            else:
                ratio = failed['ratio'].min()
            final_entries.append({'dataset':dataset,'curve':curve,'error':1-ratio, 'error_type':"undershoot"})
        for dataset_curve in error_bigger_than_1.groupby(['dataset','curve']):
            dataset, curve = dataset_curve[0]
            curr_df = dataset_curve[1]
            failed = curr_df.loc[~curr_df['is_dead']]
            if len(failed) == 0:
                ratio = 1
            else:
                ratio = failed['ratio'].max()
            final_entries.append({'dataset':dataset,'curve':curve,'error':ratio-1, 'error_type':"overshoot"})

        current_res = pd.DataFrame(final_entries)
        final_final_entries = []
        for dataset_curve in current_res.groupby(['dataset','curve']):
            dataset, curve = dataset_curve[0]
            curr_df = dataset_curve[1]
            temp=curr_df.sort_values('error',ascending=False).iloc[0:1]
            final_final_entries.append(temp)
        current_res = pd.concat(final_final_entries)

        current_res.to_csv(f'results_{self.counter}.csv')
        self.log('prediction_error',current_res['error'].mean(), on_epoch=True)
        print('here: ', current_res['error'].mean(), current_res['error'].std())
        self.results = []
        self.counter = self.counter + 1
        return super().validation_epoch_end(outputs)

    def weighted_mse_loss(self,input,target,weight):
        return torch.mean(weight*((input- target)**2))

    
    def configure_optimizers(self):
        '''defines model optimizer'''
        opt = Adam(self.parameters(), lr=self.lr)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=self.patience_lr_plateau)
        return {"optimizer": opt, "lr_scheduler":lr_schedulers, "monitor":"train_loss"}


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    