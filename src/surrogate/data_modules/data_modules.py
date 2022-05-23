# Pytorch modules
from bdb import Breakpoint
import torch
from torch.utils.data import DataLoader,Dataset, random_split

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule
from scipy.io import loadmat
# Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import hydra
import json
import random
import pickle
import os
from scipy.interpolate import interp1d
import pandas as pd
import scipy.io
from pathlib import Path

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    ## get sequence lengths
    length_current = max([ t["current"].shape[0] for t in batch ])
    lengths_voltage = max([ t["voltage"].shape[0] for t in batch ])
    lengths_xx = max([ t["xx"].shape[0] for t in batch ])
    lengths_yy = max([ t["yy"].shape[0] for t in batch ])
    lengths_tt = max([ t["tt"].shape[0] for t in batch ])

    # Current and Voltage section
    assert length_current == lengths_voltage

    # Pad current and voltage with zeros
    padded_current = torch.tensor(np.array([np.pad(x["current"], (0, length_current- x["current"].shape[0])) for i, x in enumerate(batch)]))
    padded_voltage = torch.tensor(np.array([np.pad(x["voltage"], (0, lengths_voltage - x["voltage"].shape[0])) for i, x in enumerate(batch)]))


    # # xx and yy section
    lengths = lengths_xx
    assert lengths_xx == lengths_yy and lengths_xx == lengths_tt
    padded_xx = torch.tensor(np.array([np.pad(x["xx"], (0, lengths_xx - x["xx"].shape[0])) for i, x in enumerate(batch)]))
    padded_yy = torch.tensor(np.array([np.pad(x["yy"], (0, lengths_yy - x["yy"].shape[0])) for i, x in enumerate(batch)]))
    padded_tt = torch.tensor(np.array([np.pad(x["tt"], (0, lengths_tt - x["tt"].shape[0])) for i, x in enumerate(batch)]))


    # Ugly hack, if the element in the batch contains the keys curves and dataset, means we are in a validation/test batch
    if "curve" in batch[0] and "dataset" in batch[0]:
        metadata = {
            "curve": [x["curve"] for x in batch],
            "dataset": [x["dataset"] for x in batch],
            "ratio": [x["ratio"] for x in batch],
        }
        # Even uglier hack, to include Q and R for validation/test
        if "q" in batch[0] and "r" in batch[0]:
            tmp = {
            "q": [x["q"] for x in batch],
            "r": [x["r"] for x in batch],
            }
            metadata = {**metadata, **tmp}
        if "gt_length" in batch[0]:
            tmp = {
                "gt_length": [x["gt_length"] for x in batch]
                }
            metadata = {**metadata, **tmp}
        return padded_current.float(), padded_voltage.float(), padded_xx.float(), padded_yy.float(), padded_tt.float(), metadata
    else:
        return padded_current.float(), padded_voltage.float(), padded_xx.float(), padded_yy.float(), padded_tt.float()


DATA_PATH = Path(hydra.utils.to_absolute_path('data/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab'))
def load_curves(datasets):
    curves = []
    real_datasets = datasets
    for dataset in real_datasets:
        p = DATA_PATH / dataset
        raw_data = scipy.io.loadmat(p)['data'][0][0]
        numerical_data = raw_data[0][0]
        reference_discharges = [x for x in numerical_data if x[0] == 'reference discharge']
        for idx, reference_discharges in enumerate(reference_discharges):
            #if idx % 5 == 0:
            curve = {}
            ttt = reference_discharges[2][0]
            vvv = reference_discharges[4][0]
            iii = reference_discharges[5][0]
            time_relative= np.concatenate([np.array([0]),np.cumsum(np.diff(ttt))])
            f_v = interp1d(time_relative,vvv, kind='cubic')
            f_i = interp1d(time_relative,iii, kind='cubic')
            time_new = np.arange(0,max(time_relative),2)
            voltage_new = f_v(time_new)
            current_new = f_i(time_new)
            curve['voltage'] = voltage_new
            curve['current'] = np.ones(current_new.shape)
            curve['time'] = time_new
            curve['curve'] = idx
            curve['dataset'] = dataset
            curves.append(curve)
    curves = np.array(curves)
    return curves


class RealDataset(Dataset):

    def __init__(self,datasets=['RW9.mat','RW10.mat','RW11.mat','RW12.mat'], curves=[], mode="train", requires_normalization=False):
        assert datasets != curves, "Must provide either datasets or curves, not both"

        if not curves:
            self.curves = load_curves(datasets)
        else:
            if datasets:
                print("Warning datasets argument is ignored")
            self.curves = curves

        self.mode = mode
        self.requires_normalization = requires_normalization


    def __len__(self):
        # if self.mode == "train":
        #     return len(self.curves) * 20
        # else:
        if self.mode == "train":
            return len(self.curves) * 40
        else:
            return len(self.curves) * 90

    def __getitem__(self, idx):
        idx_curve = idx%len(self.curves)
        curve = self.curves[idx_curve]
        current = curve['current']
        voltage = curve['voltage']
        
        if self.requires_normalization:
            # Apply MinMaxScaler to both voltage and current
            voltage = (voltage - 3.5)/2
            current = (current - 2.5)/2

        time = curve['time']
        
        # Five different plots
        if self.mode == "train":
            ratio = (idx // len(self.curves) +  80)/100
        else:
            ratio = (idx // len(self.curves) + 55)/100
        # if ratio > 0.8:
        gt_length = voltage.shape[0]
        #cut_off_idx = np.where(voltage<=3.2)[0][0]
        if ratio > 1:
            #assert not self.mode == "train"
            tmp = ratio-1
            voltage = np.concatenate([voltage,np.zeros(int(len(voltage)*tmp))])
        else:
            voltage = voltage[:int(ratio*len(voltage))]
        # print(self.mode)
        # print(current[0])
        if voltage.shape[0] > current.shape[0]:
            current = np.concatenate([current,np.ones(len(voltage)-len(current))])
        else:
            current = current[:len(voltage)]
        assert len(current) == len(voltage)
        datapoint = {}
        # datapoint['current'] = current
        # datapoint['voltage'] = voltage
        # Preparing Context
        length = 200
        # if self.mode == "train":
        #     x_init = random.randint(0,50)
        # else:
        x_init = 48
        #x_init=0
        datapoint['current'] = current[x_init:]
        datapoint['voltage'] = voltage[x_init:]
        xx, yy,tt = current[x_init:x_init+length], voltage[x_init:x_init+length], time[x_init:x_init+length]
        assert len(xx) == len(yy) == len(tt)
        datapoint['xx'] = xx
        datapoint['yy'] = yy
        datapoint['tt'] = tt
        datapoint['curve'] = curve['curve']
        datapoint['dataset'] = curve['dataset']
        datapoint['ratio'] = ratio
        datapoint['gt_length'] = gt_length
        return datapoint   #### be careful here: cropping current to voltage value---> need padding


class RealDatasetSingleQuery(Dataset):
    """
    Not used
    """

    def __init__(self,datasets=['RW9.mat','RW10.mat','RW11.mat','RW12.mat'], curves=[], mode="train"):
        assert datasets != curves, "Must provide either datasets or curves, not both"

        if not curves:
            self.curves = load_curves(datasets)
        else:
            self.curves = curves

        self.mode = mode
        self.total_numer_of_points = int(sum([len(x['voltage'])*0.4 for x in self.curves]))
        self.cumsum_length = np.cumsum([len(x['voltage']) for x in self.curves])

    def __len__(self):
        return self.total_numer_of_points

    def __getitem__(self, idx):

        idx_curve = np.searchsorted(self.cumsum_length,idx)
        if idx_curve == 0:
            delta_curve_point = idx
        else:
            delta_curve_point = idx - self.cumsum_length[idx_curve-1]

        curve = self.curves[idx_curve]
        current = curve['current']
        voltage = curve['voltage']
        time = curve['time']
        # Five different plots
        #ratio = (idx // len(self.curves) + 80)/100
        curve_point = int(delta_curve_point + len(voltage) * 0.8)
        if curve_point > len(voltage):
            voltage = np.concatenate([voltage,np.zeros(curve_point)])
        else:
            voltage = voltage[:curve_point]
        # else:
        #     breakpoint()
        # # if ratio > 0.8:
        # #     breakpoint()
        # breakpoint()
        # if ratio > 1:
        #     #assert not self.mode == "train"
        #     tmp = ratio-1
        #     voltage = np.concatenate([voltage,np.zeros(int(len(voltage)*tmp))])
        # else:

        if voltage.shape[0] > current.shape[0]:
            current = np.concatenate([current,np.ones(len(voltage)-len(current))])
        else:
            current = current[:len(voltage)]
        assert len(current) == len(voltage)
        datapoint = {}
        # datapoint['current'] = current
        # datapoint['voltage'] = voltage
        # Preparing Context

        length = 200
        # if self.mode == "train":
        #     x_init = random.randint(0,50)
        # else:
        x_init = 48
        #x_init=0
        datapoint['current'] = current[x_init:]
        datapoint['voltage'] = voltage[x_init:]
        xx, yy,tt = current[x_init:x_init+length], voltage[x_init:x_init+length], time[x_init:x_init+length]
        assert len(xx) == len(yy) == len(tt)
        datapoint['xx'] = xx
        datapoint['yy'] = yy
        datapoint['tt'] = tt
        datapoint['curve'] = curve['curve']
        datapoint['dataset'] = curve['dataset']

        #datapoint['ratio'] = ratio
        #breakpoint()
        return datapoint   #### be careful here: cropping current to voltage value---> need padding




class BaseDataset(Dataset):
    def __init__(self, metadata, mode, requires_normalization = False, min_init=None, max_init=None, min_length=None, max_length=None,drop_final=None, is_single_query=False):
        self.mode = mode
        self.metadata = metadata
        #self.model_type = model_type
        self.requires_normalization = requires_normalization
        # if requires_normalization:
        #     self.scaler_dict = json.load(open(metadata["data_dir"] / 'scalers.json'))

        self.max_init = max_init
        self.min_init = min_init
        self.max_length = max_length
        self.min_length = min_length
        self.drop_final=drop_final
        self.is_single_query = is_single_query
        if self.mode == "test":
            self.curves = self.metadata["test_times"]
        elif self.mode == "train":
            self.curves = self.metadata["train_times"]
        else:
            raise KeyError()


    def __len__(self):
        if self.mode == "test":
            return len(self.metadata["test_times"])*60 # Same thing as the RealDataset
        elif self.mode == "train":
            return len(self.metadata["train_times"])
        else:
            raise KeyError()



def load_dataset(data_dir, split_val=False):
    assert os.path.exists(data_dir / 'metadata.json')
    with open(data_dir.joinpath('metadata.json'), 'r') as f:
        metadata = json.load(f)

    if split_val:
        train_metadata = {}
        train_metadata["train_times"] = [x for x in range(metadata["train_times"]) if x % 10 != 0]
        train_metadata["chunk_size"] = metadata["chunk_size"]
        train_metadata["data_dir"] = data_dir

        validation_metadata = {}
        validation_metadata["train_times"] = [x for x in range(metadata["train_times"]) if x % 10 == 0]
        validation_metadata["chunk_size"] = metadata["chunk_size"]
        validation_metadata["data_dir"] = data_dir
        return train_metadata, validation_metadata
    else:
        train_metadata = {}
        train_metadata["train_times"] = [x for x in range(metadata["train_times"]) ]
        train_metadata["chunk_size"] = metadata["chunk_size"]
        train_metadata["data_dir"] = data_dir
        return train_metadata

def load_testset(data_dir, percent_val=1):
    assert os.path.exists(data_dir / 'test_metadata.json')
    with open(data_dir.joinpath('test_metadata.json'), 'r') as f:
        metadata = json.load(f)

    # We call them
    test_metadata = {}
    test_metadata["test_times"] = [x for x in range(metadata["test_times"]) if x % int(1/percent_val) == 0]
    test_metadata["chunk_size"] = metadata["chunk_size"]
    test_metadata["data_dir"] = data_dir
    return test_metadata

    #return {"voltage": voltage, "current": current, "time": time}
class DataModuleExpII(LightningDataModule):

    def __init__(self,data_dir='./data/N2300_50_sensors_100_output/', batch_size=256,num_w=24, requires_normalization=False, is_single_query=False, min_init=0, max_init=0, min_length=0, max_length=0,drop_final=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_w = num_w
        self.requires_normalization = requires_normalization
        self.is_single_query = is_single_query
        self.min_init = min_init
        self.max_init = max_init
        self.min_length = min_length
        self.max_length = max_length
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_w = num_w
        train_metadata, validation_metadata= load_dataset(data_dir, split_val=True)
        #elf.vp=vp # % of validation data
        # Only datasets with metadata are supported


            # if not is_single_query:
            #     self.training_dataset = ChunksDataset(train_metadata, mode="train", requires_normalization=requires_normalization, min_init=min_init, max_init=max_init, min_lenght=min_lenght, max_lenght=max_lenght)
            #     self.val_dataset = ChunksDataset(validation_metadata, mode="val", requires_normalization=requires_normalization, min_init=min_init, max_init=max_init, min_lenght=min_lenght, max_lenght=max_lenght)
            #     self.real_dataset = RealDataset()
            # else:
        self.training_dataset = ChunksDataset(train_metadata, mode="train", requires_normalization=requires_normalization, min_init=min_init, max_init=max_init, min_length=min_length, max_length=max_length,drop_final=drop_final, is_single_query=is_single_query)
        #self.val_dataset = ChunksDataset(validation_metadata, mode="val", requires_normalization=requires_normalization, min_init=min_init, max_init=max_init, min_length=min_length, max_length=max_length,drop_final=drop_final)
        self.real_dataset = RealDataset(requires_normalization=requires_normalization) #RealDatasetSingleQuery()


    def setup(self, stage=None):
        """called one each GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:
            self.shuffle=True
            self.drop_last=False

            # self.test_dataset = CustomTensorDataset_v2(
            #     data=[self.scaled_voltage, self.scaled_current],
            #     mode="test"
            # )
        else:
            self.shuffle=False
            self.drop_last=False



    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_w,
            collate_fn=collate_fn_padd,
        )
        return trainloader
    def val_dataloader(self):
        """returns validation dataloader"""
        valloader_real = torch.utils.data.DataLoader(
            self.real_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_w,
            collate_fn=collate_fn_padd,
        )
        # valloader_syn = torch.utils.data.DataLoader(
        #     self.val_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     drop_last=False,
        #     num_workers=self.num_w,
        #     collate_fn=self.collate_fn_padd,
        # )
        return (valloader_real,)

    def test_dataloader(self):
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_w,
            collate_fn=collate_fn_padd,
        )
        return testloader

class DONDataModule(LightningDataModule):

    def __init__(self,data_dir='./data/N2300_50_sensors_100_output/', batch_size=256,num_w=24,vp=0.15, expected_states=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_w = num_w
        self.vp=vp
        self.x= np.load(self.data_dir / 'training_states_and_observable.pkl', allow_pickle=True)
        if self.x.shape[-1] < expected_states:
            raise ValueError('Expected states is smaller than current states')
        elif expected_states == 9:
            self.x = self.x[:,:,:expected_states]
        elif expected_states == 8:
            self.x = self.x[:,:,:expected_states]
        elif expected_states == 1:
            self.x = self.x[:,:,-1:]
        else:
            raise ValueError('Expected states is not supported')
        self.expected_states = expected_states

        self.y= np.load(self.data_dir / 'training_voltages.pkl',  allow_pickle=True)
        self.t= np.load(self.data_dir / 'training_times.pkl',  allow_pickle=True)
        curr= np.load(self.data_dir / 'training_currents.pkl',  allow_pickle=True)

        self.x_t= np.load(self.data_dir / 'test_states_and_observable.pkl', allow_pickle=True)
        self.y_t= np.load(self.data_dir / 'test_voltages.pkl',  allow_pickle=True)
        self.t_t= np.load(self.data_dir / 'test_times.pkl',  allow_pickle=True)
        curr_t= np.load(self.data_dir / 'test_currents.pkl',  allow_pickle=True)

        if self.x_t.shape[-1] < expected_states:
            raise ValueError('Expected states is smaller than current states')
        elif expected_states == 9:
            self.x_t = self.x_t[:,:,:expected_states]
        elif expected_states == 8:
            self.x_t = self.x_t[:,:,:expected_states]
        elif expected_states == 1:
            self.x_t = self.x_t[:,:,-1:]
        else:
            raise ValueError('Expected states is not supported')

        #assert not (np.isclose(np.mean(self.x),np.mean(self.x_t)))
        #assert not (np.isclose(np.mean(self.y),np.mean(self.y_t)))
        #assert not (np.isclose(np.mean(self.t),np.mean(self.t_t)))
        # Convert current obj to current array, by naively concatenating for each current obj the current values and times
        self.curr = np.array([np.array(x[0].return_current_profile_array(time_denominator=1000,pad_profiles=True)) for x in curr])
        self.curr_t =    np.array([np.array(x[0].return_current_profile_array(time_denominator=1000,pad_profiles=True)) for x in curr_t])
        print("Current shape: ",self.curr.shape)


    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)

        self.train_dataset,self.test_dataset=self.normalization()
        if stage == 'fit' or stage is None:
            pp=round(self.vp*len(self.train_dataset)) # number of validation samples
            self.train, self.val = random_split(self.train_dataset, [len(self.train_dataset)-pp, pp],generator=torch.Generator().manual_seed(42))
        if stage == 'test' or stage is None:
            self.test = self.test_dataset

    def train_dataloader(self):
        '''returns training dataloader'''
        data_train = DataLoader(self.train, batch_size=self.batch_size,num_workers=self.num_w)
        return data_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        mnist_val1 = DataLoader(self.val, batch_size=self.batch_size,num_workers=self.num_w)
        mnist_val2 = DataLoader(self.test_dataset, batch_size=len(self.test_dataset),num_workers=self.num_w)
        return [mnist_val1,mnist_val2]

    def test_dataloader(self):
        '''returns test dataloader'''
        mnist_test = DataLoader(self.test, batch_size=self.batch_size,num_workers=self.num_w)
        return mnist_test


class TestDataloader():
    def __init__(self, num_w=24, init_value=0, length=200, requires_normalization=False, experiment_type=None, percent_val=1, cfg=None):
        super().__init__()
        self.num_w = num_w

        self.test_dataset = {}
        if experiment_type == "constant":
            test_constant_all = load_testset(Path(hydra.utils.to_absolute_path("/local/home/lbiggio/battery_proj/test_dataset/07-03-36/data")), percent_val)
            self.test_dataset["test_constant_all"] = ChunksDataset(test_constant_all,"test",  min_init=init_value, max_init=init_value, min_length=length, max_length=length, requires_normalization=requires_normalization)
            
            #self.test_dataset["real"] = RealDataset() #RealDatasetSingleQuery()

        elif experiment_type == "variable":
            test_variable_10 = load_testset(Path(hydra.utils.to_absolute_path(cfg.test_path_variable)), percent_val)
            self.test_dataset["val_variable_0"] = ChunksDataset(test_variable_10, "test", min_init=init_value, max_init=init_value, min_length=length, max_length=length, requires_normalization=requires_normalization)

        else:
            raise KeyError()


    def test_dataloader(self):
        """returns validation dataloader"""

        self.test_dataloaders = {}
        for key, dataset in self.test_dataset.items():
            self.test_dataloaders[key] = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                collate_fn=collate_fn_padd,
            )
        return self.test_dataloaders

class FinetuningTestDataloader():
    def __init__(self, test_curves, num_w=24, cfg=None):
        super().__init__()
        self.num_w = num_w

        self.test_dataset = {}
        self.test_dataset["test_finetuning"] = RealDataset(datasets=[], curves=test_curves, mode="test", requires_normalization=False )
            
    def test_dataloader(self):
        """returns validation dataloader"""

        self.test_dataloaders = {}
        for key, dataset in self.test_dataset.items():
            self.test_dataloaders[key] = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                collate_fn=collate_fn_padd,
            )
        return self.test_dataloaders

class DataModuleExpII_finetuning(DataModuleExpII):

    def __init__(self,train_curves,val_curves,batch_size=256,num_w=24, requires_normalization=False, is_single_query=False):
        #super().__init__()
        LightningDataModule.__init__(self)
        self.batch_size = batch_size
        self.num_w = num_w
        self.training_dataset = RealDataset(curves=train_curves, mode="train")
        self.real_dataset = RealDataset(curves=val_curves, mode="val")
        #self.test_dataset = RealDataset(datasets=test_datasets)

class ChunksDataset(BaseDataset):
    def __getitem__(self, idx):
        #self.metadata["data_dir"][idx]
        while True:
            if self.mode == "test":
                idx_curve = idx%len(self.curves)
                # curve = self.metadata["test_times"][idx_curve]
                # index = self.metadata["test_times"][idx]
                # file_idx = (index // self.metadata["chunk_size"]) * self.metadata["chunk_size"]
                # sample_idx = index % self.metadata["chunk_size"]
            # elif self.mode == "train":
            elif self.mode == "train":
                idx_curve = idx
                
            else:
                raise KeyError("mode must be either 'train' or 'test'")
            index = self.metadata[f"{self.mode}_times"][idx_curve]
            file_idx = (index // self.metadata["chunk_size"]) * self.metadata["chunk_size"]
            sample_idx = index % self.metadata["chunk_size"]
            # file_idx = 1920
            # sample_idx = 4
            # import pdb
            # pdb.set_trace()

            # else:
            #     raise KeyError()
            #print("file_idx: ", file_idx, "sample_idx: ", sample_idx)
            # Load train current and voltage
            #current =
            if self.mode == "test":
                current_path = self.metadata["data_dir"] / f"{self.mode}_currentss_{file_idx}.pkl"
            else:
                current_path = self.metadata["data_dir"] / f"train_currents_{file_idx}.pkl"
            voltage_path = self.metadata["data_dir"] / f"{self.mode}_voltages_{file_idx}.pkl"
            times_path = self.metadata["data_dir"] / f"{self.mode}_times_{file_idx}.pkl"
            q_path = self.metadata["data_dir"] / f"{self.mode}_Qs_{file_idx}.pkl"
            r_path = self.metadata["data_dir"] / f"{self.mode}_Rs_{file_idx}.pkl"
            # elif self.mode == "train":
            #     
            #     voltage_path = self.metadata["data_dir"] / f"train_voltages_{file_idx}.pkl"
            #     times_path = self.metadata["data_dir"] / f"train_times_{file_idx}.pkl"
            # else:
            #     raise KeyError()

            with open(current_path, 'rb') as f:
                current_batch = pickle.load(f)

            with open(voltage_path, 'rb') as f:
                voltage_batch = pickle.load(f)

            with open(times_path, 'rb') as f:
                times_batch = pickle.load(f)

            if self.mode == "test":
                with open(q_path, 'rb') as f:
                    q_batch = pickle.load(f)

                with open(r_path, 'rb') as f:
                    r_batch = pickle.load(f)

            
            voltage = voltage_batch[sample_idx]
            if self.mode == "train":
                # First index where voltage is lower than 3.2
                cut_off_idx = np.where(voltage<=3.2)[0][0]
                
                if cut_off_idx < 300:
                    # Sample another 
                    idx = random.randint(0,len(self.metadata["train_times"])-1)
                    continue
            break
        current = current_batch[sample_idx]
        voltage = voltage
        times = np.array(times_batch[sample_idx])
        current = np.array(current.get_current_profile(len(voltage)*2)) #TODO: FIXME when currents are variable #[:max_length]

        assert len(voltage) == len(current) == len(times)

        # Apply scalers if required
        if self.requires_normalization:
            # Apply MinMaxScaler to both voltage and current
            voltage = (voltage - 3.5)/2
            current = (current - 2.5)/2
            # voltage_std = (voltage - self.scaler_dict["voltage"][0]) / (self.scaler_dict["voltage"][1] - self.scaler_dict["voltage"][0])
        cut_off_idx = np.where(voltage<=3.2)[0][0]
        if self.mode == "test":
            ratio = (idx // len(self.curves) + 70)/100
            voltage = voltage[:cut_off_idx]
            gt_lenght = len(voltage)
            current = current[:len(voltage)]
            if ratio > 1:
                #assert not self.mode == "train"
                tmp = ratio-1
                voltage = np.concatenate([voltage,np.zeros(int(len(voltage)*tmp))])
            else:
                voltage = voltage[:int(ratio*len(voltage))]
            if voltage.shape[0] > current.shape[0]:
                last_current_val = current[-1]
                current = np.concatenate([current,last_current_val*np.ones(len(voltage)-len(current))])
            else:
                current = current[:len(voltage)]
        elif self.mode == "train":
            
            #swapped because overwritten
            extendable_current = current[cut_off_idx:]
            extendable_voltage = voltage[cut_off_idx:]
            current = current[:cut_off_idx]
            voltage = voltage[:cut_off_idx]
            # Make sure that that the trajectory is longer at least 300 points considering ratio
            min_ratio = int(300 / len(voltage) * 100)
            if not self.is_single_query:    
                ratio = random.randint(max(min_ratio,55),160)/100
            else:
                ratio = 1.6 # Always use the maximum ratio
            if ratio > 1:
                #assert not self.mode == "train"
                tmp = ratio-1
                concat_len = int(len(voltage)*tmp)
                to_add_voltage = extendable_voltage[:concat_len]
                to_add_current = extendable_current[:concat_len]
                voltage = np.concatenate([voltage,to_add_voltage])
                current = np.concatenate([current,to_add_current])
            else:
                new_len = int(len(voltage)*ratio)
                voltage = voltage[:new_len]
                current = current[:new_len]

            # if self.drop_final:
            #     ratio = random.uniform(0.7,1) #random.randint(0,250)
            #     new_len = max(500, int(len(voltage)*ratio))
            # else:
            #     ratio=1 #FIXME when we have variable length and fnn and deeponet, we end up here. Not sure if it correct.

            #new_len = max(500, int(len(voltage)*ratio))
            # voltage = voltage[:new_len]
            # current = current[:len(voltage)]
            assert len(current) == len(voltage)
        else:
            raise KeyError()
        #scaled_current = scaled_current[:max_length]

        max_length = self.max_length
        min_length = self.min_length
        if max_length == min_length:
            length = max_length
        else:
            length = np.random.randint(min_length, max_length)

        min_init = self.min_init
        max_init = min(self.max_init,max(len(current)- length,0))
        if max_init <= min_init:
            x_init = max_init
        else:
            x_init = np.random.randint(min_init, max_init)
        xx, yy,tt = current[x_init:x_init+length], voltage[x_init:x_init+length], times[x_init:x_init+length]
        assert len(xx) == len(yy) == len(tt)
        # if len(xx) == 0:
        #     breakpoint()

        datapoint = {}
        datapoint['current'] = current[x_init:]
        datapoint['voltage'] = voltage[x_init:]

        datapoint['xx'] = xx
        datapoint['yy'] = yy
        datapoint['tt'] = tt
        if self.mode == "test":
            datapoint['ratio'] = ratio
            datapoint['curve'] = sample_idx
            datapoint['dataset'] = file_idx
            datapoint['q'] = q_batch[sample_idx]
            datapoint['r'] = r_batch[sample_idx]  
            datapoint['gt_length'] = gt_lenght  

        # if (yy == 0).all():
        #     breakpoint()
        return datapoint   #### be careful here: cropping current to voltage value---> need padding

