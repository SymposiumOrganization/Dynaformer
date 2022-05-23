import os
import omegaconf

interpolation_region_q = ((5000,8000),)
extrapolation_region_q = ((4000,5000),(8000,9000))
interpolation_region_r = ((0.017215,0.45),)
extrapolation_region_r = ((0.015,0.017215),(0.45,0.5))

def filter_all_temporal_error_csv(df):
    bool_all_q = (extrapolation_region_q[0][0] <= df['q']) & (df['q'] <= extrapolation_region_q[1][1])
    bool_all_r = (extrapolation_region_r[0][0] <= df['r']) & (df['r'] <= extrapolation_region_r[1][1])
    bool_all = bool_all_q & bool_all_r
    return df.loc[bool_all]

def filter_interpolation_temporal_error_csv(df):
    bool_interpolation_q = df['q'].between(*interpolation_region_q[0])
    bool_interpolation_r = df['r'].between(*interpolation_region_r[0])
    bool_interpolation = bool_interpolation_q & bool_interpolation_r
    return df.loc[bool_interpolation]

def filter_extrapolation_temporal_error_csv(df):
    bool_q = (df['q'].between(*extrapolation_region_q[0])) | (df['q'].between(*extrapolation_region_q[1]))
    bool_r = (df['r'].between(*extrapolation_region_r[0])) | (df['r'].between(*extrapolation_region_r[1]))
    bool_extrapolation = bool_q | bool_r
    return df.loc[bool_extrapolation]

def set_cuda_visible_device(cfg):
    if type(cfg.cuda_visible_devices) == int:
        cfg.cuda_visible_devices = str(cfg.cuda_visible_devices)
   
    if type(cfg.cuda_visible_devices) == str:
        os.environ["CUDA_VISIBLE_DEVICES"]=cfg.cuda_visible_devices
        gpus = 1
    elif type(cfg.cuda_visible_devices) == omegaconf.listconfig.ListConfig:
        gpus = len(cfg.cuda_visible_devices)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfg.cuda_visible_devices])
    else:
        gpus = -1

    return gpus 