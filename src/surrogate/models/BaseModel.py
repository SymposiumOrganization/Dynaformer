import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule

import numpy as np
import pandas as pd
from collections import defaultdict
import pdb


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

class BaseModel(LightningModule):
    """Base class for all models, contains common functions such as training_step, validation_step, etc."""
    def __init__(self, requires_normalization=False, variable_current=False):
        super().__init__()
        self.results = []
        self.counter = 0
        self.requires_normalization = requires_normalization
        self.variable_current = variable_current


        
        #weights = torch.ones(weights.shape,device=self.device)
#        weights = weights[zero_mask]
#        out = out[zero_mask].squeeze()
#        vol = padded_voltage[zero_mask].squeeze()
#        loss = self.weighted_mse_loss(out,vol,weights)
    def current_to_embedding(self, curr_padded, curr_zero_mask):
        unpad_curr=curr_padded[curr_zero_mask]
        unique_currents=torch.unique_consecutive(unpad_curr)
        padding_emb = torch.zeros(10-len(unique_currents),device=self.device)
        curr_times = torch.cat([torch.where(torch.diff(unpad_curr))[0],torch.tensor([unpad_curr.shape[0]],device=self.device)])/1000
        query_curr_temp=torch.cat([unique_currents,padding_emb,curr_times,padding_emb])
        assert len(unique_currents)==len(curr_times)
        return query_curr_temp

    def training_step(self, input, batch_idx):
        '''needs to return a loss from a single batch'''
        padded_current, padded_voltage, padded_xx, padded_yy, padded_tt= input
        inp = torch.cat([padded_xx.unsqueeze(2),padded_yy.unsqueeze(2),padded_tt.unsqueeze(2) / 1000],axis=2)
        # Randomly select the support points 
        zero_mask = padded_voltage != 0 # zero only for padding
        query_curr = [] #torch.zeros([1,20], device=self.device)
        query_times = []
        targets = []
        W=[]
        factor = 0.01
        weights = torch.arange(0,(padded_voltage.shape[1]-0.5)*factor,factor,device=self.device).repeat(input[0].shape[0],1) + 1
        for i in range(padded_current.shape[0]):
            curr_padded = padded_current[i,:]
            curr_zero_mask = zero_mask[i,:]
            if self.variable_current:
                curr_query_curr = current_to_embedding(curr_padded, curr_zero_mask)
                query_curr.append(curr_query_curr)
            idx_temp = np.random.choice(len(curr_padded[curr_zero_mask]), 1, replace=False)[0] * 2
            idx = int(padded_tt[0][0].item())+idx_temp
            query_times.append(idx)
            targets.append(padded_voltage[i,idx_temp//2])
            W.append(weights[i,idx_temp//2])
            #idx = np.random.choice(len(padded_current[i,:]), 1, replace=False)[0]
            #padded_current[i,idx] = 0
        query_times = torch.tensor(query_times, device=self.device) / 1000
        if self.variable_current:
            query_curr = torch.cat(query_curr,axis=0)
            #query_curr = torch.tensor(query_curr[1:], device=self.device)
            pred_voltages = self.forward(inp, query_curr, query_times.unsqueeze(1))
        else:
            pred_voltages = self.forward(inp, padded_current[:,0:1], query_times[:,None])
        targets = torch.tensor(targets, device=self.device)
        weights = torch.tensor(W, device=self.device)
        if self.cfg.loss == "rmse":
            loss = self.loss_func(pred_voltages.squeeze(), targets.squeeze())
        else:
            loss = self.weighted_mse_loss(pred_voltages.squeeze(), targets,weights)
        # Log training loss
        self.log('train_loss', loss,on_step=False,on_epoch=True)
        return loss

    def validation_step(self, input, batch_idx):    
        '''used for logging metrics'''
        #batch, lengths, mask = input
        val_idx=0
        if val_idx == 0:
            padded_current, padded_voltage, padded_xx, padded_yy, padded_tt, metadata= input
        else:
            padded_current, padded_voltage, padded_xx, padded_yy, padded_tt= input

        #xx, yy, x, y = preprocessing(batch)
        inp = torch.cat([padded_xx.unsqueeze(2),padded_yy.unsqueeze(2),padded_tt.unsqueeze(2)/ 1000],axis=2)
        # if True:
        #     padding = torch.zeros([inp.shape[0],100,inp.shape[2]], device=self.device)
        #     inp = torch.cat([inp,padding],axis=1)
        
        zero_mask = padded_voltage != 0 # zero only for padding

        # Find the index of the last non-zero value for each row for padded_current
        query_times = []
        for i in range(padded_current.shape[0]):
            curr_padded = padded_current[i,:]
            index = int(padded_tt[0][0].item()) + len(curr_padded[zero_mask[i,:]]) * 2 - 2 # multiply by 2 because we have a time delta of 2 seconds
            query_times.append(index)
        query_times = torch.tensor(query_times, device=self.device)  / 1000
        #query_time = 
        if val_idx == 0:
            #gt_voltage = padded_voltage[padded_voltage!=0]
            pred_voltages = self.forward(inp, padded_current[:,0:1], query_times[:,None]) #Since these methods only works only for constant current itis okey
            #out1 = self.forward(inp[-1:,:,:],padded_current[-1:,:])
            if self.requires_normalization:
                pred_voltages = pred_voltages*2 + 3.5
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

        # breakpoint()
        # xx, yy, x, y = preprocessing(batch)
        # inp = torch.cat([xx,yy],axis=2)
        # # Randomly select query and support samples
        # query_time = []
        # query_y = []
        # for i in range(inp.shape[0]):
        #     idx = np.random.choice(len(y), 1, replace=False)[0]
        #     query_time.append(idx)
        #     query_y.append(y[i,idx])

        # query_time = torch.tensor(query_time)
        # query_y = torch.cat(query_y)
        # out = self([xx,yy], y, query_time)
        # loss = self.loss_func(out.squeeze(), y.squeeze()) 
        # # Log validation loss (will be automatically averaged over an epoch)
        # self.log('valid_loss',loss,on_step=False,on_epoch=True,sync_dist=True)
        # else:
        #     self.log('test_loss',loss,on_step=False,on_epoch=True,sync_dist=True)

    # def test_step(self, batch, batch_idx):
    #     '''used for logging metrics'''
    #     x, y = batch
    #     out = self(x)
    #     loss = self.loss_func(out.squeeze(), y.squeeze())

        # # Log test loss
        # self.log('test_loss', loss,sync_dist=True)
    def validation_epoch_end(self, outputs) -> None:
        df = pd.concat(self.results)
        #units = defaultdict()
        #units_new = defaultdict()
        #units_zero= defaultdict()
        #error_smaller_than_1 = df.loc[(df['ratio'] < 1) & (df['is_dead'] == True)]
        #error_bigger_than_1 = df.loc[(df['ratio'] > 1) & (df['is_dead'] == False)]
        error_smaller_than_1 = df.loc[(df['ratio'] < 1)]
        error_bigger_than_1 = df.loc[(df['ratio'] > 1)]
        #for d in np.unique(df['dataset']):
        #    units[d]=np.unique(df[df['dataset']==d]['curve'].values)
        #    units_new[d]=np.unique(np.concatenate([error_bigger_than_1[error_bigger_than_1['dataset']==d]['curve'].values,error_smaller_than_1[error_smaller_than_1['dataset']==d]['curve'].values]))
        #    units_zero[d] = np.setdiff1d(units[d],units_new[d])
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
        #for k in units_zero.keys():
        #    for u in units_zero[k]:
        #        final_entries.append({'dataset':k,'curve':u,'error':0, 'error_type':"match"})
        current_res = pd.DataFrame(final_entries)
        final_final_entries = []
        for dataset_curve in current_res.groupby(['dataset','curve']):
            dataset, curve = dataset_curve[0]
            curr_df = dataset_curve[1]
            temp=curr_df.sort_values('error',ascending=False).iloc[0:1]
            final_final_entries.append(temp)
        current_res = pd.concat(final_final_entries)
        #todelete=[]
        #for d in np.unique(df['dataset']):
        #    temp=np.where(np.unique(current_res[current_res['dataset']==d]['curve'].values,return_counts=True)[1]>1)[0]
        #    doubles = np.unique(current_res[current_res['dataset']==d]['curve'].values,return_counts=True)[0][temp]
        #    if len(temp)>=1:
        #        indeces = current_res[current_res['dataset']==d]['curve'][current_res[current_res['dataset']==d]['curve']== doubles[0]].index
        #        todelete.append(indeces[np.argmin(current_res.iloc[indeces,:]['error'])])
        #current_res=current_res.drop(todelete)
        current_res.to_csv(f'results_{self.counter}.csv')
        self.log('prediction_error',current_res['error'].mean(), on_epoch=True)
        print('here: ', current_res['error'].mean(), current_res['error'].std())
        self.results = []
        self.counter = self.counter + 1
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def weighted_mse_loss(self,input,target,weight):
        return torch.mean(weight*((input- target)**2))



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
        #units = defaultdict()
        #units_new = defaultdict()
        #units_zero= defaultdict()
        #error_smaller_than_1 = df.loc[(df['ratio'] < 1) & (df['is_dead'] == True)]
        #error_bigger_than_1 = df.loc[(df['ratio'] > 1) & (df['is_dead'] == False)]
        error_smaller_than_1 = df.loc[(df['ratio'] < 1)]
        error_bigger_than_1 = df.loc[(df['ratio'] > 1)]
        #for d in np.unique(df['dataset']):
        #    units[d]=np.unique(df[df['dataset']==d]['curve'].values)
        #    units_new[d]=np.unique(np.concatenate([error_bigger_than_1[error_bigger_than_1['dataset']==d]['curve'].values,error_smaller_than_1[error_smaller_than_1['dataset']==d]['curve'].values]))
        #    units_zero[d] = np.setdiff1d(units[d],units_new[d])
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
        #for k in units_zero.keys():
        #    for u in units_zero[k]:
        #        final_entries.append({'dataset':k,'curve':u,'error':0, 'error_type':"match"})
        current_res = pd.DataFrame(final_entries)
        final_final_entries = []
        for dataset_curve in current_res.groupby(['dataset','curve']):
            dataset, curve = dataset_curve[0]
            curr_df = dataset_curve[1]
            temp=curr_df.sort_values('error',ascending=False).iloc[0:1]
            final_final_entries.append(temp)
        current_res = pd.concat(final_final_entries)
        #todelete=[]
        #for d in np.unique(df['dataset']):
        #    temp=np.where(np.unique(current_res[current_res['dataset']==d]['curve'].values,return_counts=True)[1]>1)[0]
        #    doubles = np.unique(current_res[current_res['dataset']==d]['curve'].values,return_counts=True)[0][temp]
        #    if len(temp)>=1:
        #        indeces = current_res[current_res['dataset']==d]['curve'][current_res[current_res['dataset']==d]['curve']== doubles[0]].index
        #        todelete.append(indeces[np.argmin(current_res.iloc[indeces,:]['error'])])
        #current_res=current_res.drop(todelete)
        current_res.to_csv(f'results_{self.counter}.csv')
        self.log('prediction_error',current_res['error'].mean(), on_epoch=True)
        print('here: ', current_res['error'].mean(), current_res['error'].std())
        self.results = []
        self.counter = self.counter + 1
        return super().validation_epoch_end(outputs)

    def weighted_mse_loss(self,input,target,weight):
        return torch.mean(weight*((input- target)**2))
    #def test_step(self, input, batch_idx):
    #    padded_current, padded_voltage, padded_xx, padded_yy, padded_tt= input
    #    inp = torch.cat([padded_xx.unsqueeze(2),padded_yy.unsqueeze(2),padded_tt.unsqueeze(2)],axis=2)
    #    out = self.forward(inp,padded_current)
    #    zero_mask = padded_voltage != 0 # zero only for padding
    #    loss = self.loss_func(out[zero_mask].squeeze(), padded_voltage[zero_mask].squeeze())
    #    self.log('test_loss',loss,on_step=True,on_epoch=True)
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        opt = Adam(self.parameters(), lr=self.lr)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=self.patience_lr_plateau)
        return {"optimizer": opt, "lr_scheduler":lr_schedulers, "monitor":"train_loss"}


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    