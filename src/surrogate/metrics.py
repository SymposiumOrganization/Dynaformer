import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import torch
import pandas as pd

def compute_rmse(predictions, targets):
    """
    Compute RMSE
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))

def compute_mse(predictions, targets):
    """
    Compute MSE
    """
    return np.mean((predictions - targets) ** 2)

def compute_wasserstein_distance(predictions, targets):
    """
    Compute Wasserstein distance
    """
    return stats.wasserstein_distance(predictions, targets)

def wrapper(gen):
  while True:
    try:
      yield next(gen)
    except StopIteration:
      break
    except Exception as e:
     
      print(f"Failed {e}")
    
    

def create_report(dataloaders, model, cfg):
    res = []
    for key, dataloader in dataloaders.items():
        counter = 0
        for idx, sample in tqdm(wrapper(enumerate(dataloader)), total=len(dataloader)):
            counter = counter + 1
            #print(idx)
            curr_sample = {}
            padded_current, padded_voltage, padded_xx, padded_yy, padded_tt, metadata = sample
            assert padded_current.shape[0] == 1
            
            zero_mask = padded_voltage != 0
            # TODO: when there is padding?
            curr_unpadded_curr = padded_current[zero_mask]
            curr_unpadded_voltage = padded_voltage[zero_mask].cuda()

            # if metadata["ratio"][0] < 1.1:
            #     continue
            if metadata["ratio"][0] <0.7 or metadata["ratio"][0] > 1.3:
                continue
            if metadata["ratio"] == 1:
                # Check that voltage is almost 3.2
                assert torch.isclose(curr_unpadded_voltage[-1],torch.tensor(3.2), rtol=0.001)
            if cfg.method.name == "fnn" or cfg.method.name ==  "deepOnet":
                inp = torch.cat([padded_xx.unsqueeze(2),padded_yy.unsqueeze(2),padded_tt.unsqueeze(2) / 1000],axis=2).cuda()
                if "variable" in cfg.experiment_type:
                    curr_query_curr = model.current_to_embedding(padded_current[0].cuda(), zero_mask[0].cuda())
                elif "constant" in cfg.experiment_type:
                    curr_query_curr = padded_current[0].cuda()[-1].unsqueeze(0)
                pred_voltages = []

                inps = []
                curr_query_currs = []
                query_timess = []

                for idx_temp in range(0,len(curr_unpadded_curr)):
                    idx = int(padded_tt[0][0].item())+idx_temp*2
                    #query_times.append(idx)
                    query_times = idx / 1000 #torch.tensor([idx]) / 1000
                    
                    #inps.append(inp)
                    #curr_query_currs.append(curr_query_curr)
                    query_timess.append(query_times)
                
                # import pdb
                # pdb.set_trace()
                # Repeat inp and curr_query_curr n times, where n is the dimension of len(query_timess)
                inp = torch.cat([inp]*len(query_timess),dim=0)
                curr_query_curr = torch.cat([curr_query_curr]*len(query_timess),dim=0)

                #inp = torch.cat(inps,axis=0)
                #curr_query_curr = torch.cat(curr_query_currs,axis=0)
                query_times = torch.tensor(query_timess).cuda()
                
                pred_voltages = model.forward(inp, curr_query_curr.reshape(-1,1), query_times.reshape(-1,1))
                #pred_voltages.append(model.forward(inp, curr_query_curr.unsqueeze(0), query_times.unsqueeze(1)))
    
                #pred_voltages=torch.cat(pred_voltages)
            elif cfg.method.name == "transformer" or cfg.method.name == "s2s":
                inp = torch.cat([padded_xx.unsqueeze(2),padded_yy.unsqueeze(2),padded_tt.unsqueeze(2) ],axis=2).cuda()
                #m = time.time()
                # if key == 'real':
                #     breakpoint()
                pred_voltages = model.forward(inp.cuda(), padded_current.cuda())
                #e = time.time()
                # print("Time taken: ", m-s)
                # print("Time taken: ", e-s)
            pred_voltages = pred_voltages.squeeze()
            last_val = pred_voltages[-1]
            if metadata["ratio"][0] > 1: # Then we compute the rmse
                #ratio_to_remove = metadata["ratio"][0] -1 
                #vals_to_remove = int(len(pred_voltages)/metadata["ratio"][0] * ratio_to_remove)
                padded_pred_voltages = pred_voltages[:curr_unpadded_voltage.shape[0]]
            else:
                padded_pred_voltages = pred_voltages
              
            rmse = torch.sqrt(torch.mean((padded_pred_voltages - curr_unpadded_voltage)**2))
            curr_sample["rmse"] = float(rmse) 
            curr_sample["last_val"] = float(last_val)
            if cfg.method.name == "fnn" or cfg.method.name ==  "deepOnet": # These methods tends to go back to high values after touching 3.2v
                is_dead = bool(pred_voltages.min() < 3.2)

            elif cfg.method.name == "transformer" or cfg.method.name == "s2s":
                is_dead = curr_sample["last_val"] < 3.2

            curr_sample["is_dead"] = is_dead
            curr_sample["ratio"] = metadata["ratio"][0]
            curr_sample["gt_length"] = metadata["gt_length"][0]
            curr_sample["dataset"] = metadata["dataset"][0]
            curr_sample["curve"] = metadata["curve"][0]
            curr_sample["current"] =  float(curr_unpadded_curr[0])
            curr_sample["len"] = len(pred_voltages) # Prediction length. Doesn't include samples before x_init
            curr_sample["is_constant"] = len(np.unique(curr_unpadded_curr)) == 1
            try:
                curr_sample["q"] = metadata["q"][0]
                curr_sample["r"] = metadata["r"][0]
            except:
                curr_sample["q"] = 'unknown'
                curr_sample["r"] = 'unknown'               
            curr_sample["dataloader"] = key
            res.append(curr_sample)

    df=pd.DataFrame(res)
    df.to_csv("intermediate_results.csv")

    # Post Processing
    print("Post Processing")
    temporal_final_entries = []
    rmse_final_entries = []
    for info, curr_df in tqdm(df.groupby(["dataloader","dataset","curve"])):
        dataloader, dataset, curve = info
        assert len(set(curr_df["q"])) == 1
        assert len(set(curr_df["r"])) == 1

        # Used to get temporal error
        error_smaller_than_1 = curr_df.loc[(curr_df['ratio'] < 1)]
        error_bigger_than_1 = curr_df.loc[(curr_df['ratio'] > 1)]
        metadata = {'dataloader': dataloader, 'dataset':dataset,'curve':curve, 'gt_length': curr_df["gt_length"].iloc[0]}

        rmse_statistics = {"min_rmse": curr_df["rmse"].min(), "max_rmse": curr_df["rmse"].max(), "mean":  curr_df["rmse"].mean(), "var": curr_df["rmse"].var()}
        rmse_statistics = {**metadata, **rmse_statistics}

        rmse_final_entries.append(rmse_statistics)
        for dataset_curve in error_smaller_than_1.groupby(['dataset','curve']):
            dataset, curve = dataset_curve[0]
            small_curr_df = dataset_curve[1]
            small_curr_df.sort_values(by="ratio",inplace=True)
            failed = small_curr_df.loc[small_curr_df['is_dead']]
            if len(failed) == 0:
                entry = small_curr_df.iloc[-1]
                entry["ratio"] = 1
            else:
                entry = failed.iloc[0]
            tmp = {'error':1-entry["ratio"], 'error_type':"undershoot", 'q':entry['q'], 'r':entry['r'], 'current':entry['current'], 'length':entry['len'], 'is_constant':entry['is_constant'], 'rmse':entry['rmse']}
            tmp = {**metadata, **tmp}
            temporal_final_entries.append(tmp)
        for dataset_curve in error_bigger_than_1.groupby(['dataset','curve']):
            dataset, curve = dataset_curve[0]
            high_curr_df = dataset_curve[1]
            high_curr_df.sort_values(by="ratio",inplace=True)
            goods = high_curr_df.loc[high_curr_df['is_dead']]
            if len(goods) == 0:
                entry = high_curr_df.iloc[-1]
                entry["ratio"] = 1.29
            else:
                entry = goods.iloc[0]

            tmp = {'error':entry["ratio"]-1, 'error_type':"overshoot", 'q':entry['q'], 'r':entry['r'], 'current':entry['current'], 'length':entry['len'], 'is_constant':entry['is_constant'], 'rmse':entry['rmse']}
            tmp = {**metadata, **tmp}
            temporal_final_entries.append(tmp)

    current_res = pd.DataFrame(temporal_final_entries)
    final_final_entries = []
    for dataset_curve in current_res.groupby(['dataloader','dataset','curve']):
        dataloader, dataset, curve = dataset_curve[0]
        curr_df = dataset_curve[1]
        temp=curr_df.sort_values('error',ascending=False).iloc[0:1]
        final_final_entries.append(temp)
    current_res = pd.concat(final_final_entries)
    current_res.to_csv("temporal_error_final_results_v2.csv")

    final_res = pd.DataFrame(rmse_final_entries)
    final_res.to_csv("rmse_error_final_results_.csv")
    
    # Create a summary csv
    fin_dict_temp = {}
    fin_dict_rmse = {}
    for key in dataloaders.keys():
        fin_dict_temp[key] = current_res[current_res['dataloader'] == key]["error"].mean()
        fin_dict_rmse[key] = final_res[final_res['dataloader'] == key]["mean"].mean()
    
    fin_df_temp = pd.DataFrame(fin_dict_temp, index=[0])
    fin_df_rmse = pd.DataFrame(fin_dict_rmse, index=[0])
    fin_df = pd.concat([fin_df_temp, fin_df_rmse], axis=1)
    fin_df.to_csv("summary.csv")
