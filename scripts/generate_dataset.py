import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy
import pickle
from prog_models.models import BatteryElectroChemEOD
import hydra
from tqdm import tqdm
import itertools
from pathlib import Path
from dynaformer import CurrentProfile
from functools import partial
import json

def return_entries(selected, times, voltage, currents,Q,R):
    """
    Returns the entries specified in selected for voltage, times and states
    """
    selected_time = [times[i] for i in selected]
    selected_voltage = [voltage[i] for i in selected]
    current_values = [currents[i] for i in selected]
    Q = [Q[i] for i in selected]
    R = [R[i] for i in selected]
    return selected_time, selected_voltage, current_values, Q, R


def update(ps):
    """
    Updates the battery model with the new parameters (This will be deprecated when the github repo is updated)
    """
    ps['qMax'] = ps['qMobile']/(ps['xnMax']-ps['xnMin']) # note qMax = qn+qp
    # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
    # same and the surface/bulk split is the same for both electrodes
    ps['VolS'] = ps['VolSFraction']*ps['Vol'] # surface volume
    ps['VolB'] = ps['Vol'] - ps['VolS'] # bulk volume

    # set up charges (Li ions)
    ps['qpMin'] = ps['qMax']*ps['xpMin'] # min charge at pos electrode
    ps['qpMax'] = ps['qMax']*ps['xpMax'] # max charge at pos electrode
    ps['qpSMin'] = ps['qMax']*ps['xpMin']*ps['VolSFraction'] # min charge at surface, pos electrode
    ps['qpBMin'] = ps['qMax']*ps['xpMin']*(ps['Vol'] - ps['VolS'])/ps['Vol'] # min charge at bulk, pos electrode
    ps['qpSMax'] = ps['qMax']*ps['xpMax']*ps['VolS']/ps['Vol'] # max charge at surface, pos electrode
    ps['qpBMax'] = ps['qMax']*ps['xpMax']*ps['VolB']/ps['Vol'] # max charge at bulk, pos electrode
    ps['qnMin'] = ps['qMax']*ps['xnMin'] # max charge at neg electrode
    ps['qnMax'] = ps['qMax']*ps['xnMax'] # max charge at neg electrode
    ps['qnSMax'] = ps['qMax']*ps['xnMax']*ps['VolSFraction'] # max charge at surface, neg electrode
    ps['qnBMax'] = ps['qMax']*ps['xnMax']*(1-ps['VolSFraction']) # max charge at bulk, neg electrode
    ps['qnSMin'] = ps['qMax']*ps['xnMin']*ps['VolSFraction'] # min charge at surface, neg electrode
    ps['qnBMin'] = ps['qMax']*ps['xnMin']*(1-ps['VolSFraction']) # min charge at bulk, neg electrode
    ps['qSMax'] = ps['qMax']*ps['VolSFraction'] # max charge at surface (pos and neg)
    ps['qBMax'] = ps['qMax']*(1-ps['VolSFraction']) # max charge at bulk (pos and neg)
    ps['x0']['qpS'] = ps['qpSMin']
    ps['x0']['qpB'] = ps['qpBMin']
    ps['x0']['qnS'] = ps['qnSMax']
    ps['x0']['qnB'] = ps['qnBMax']
    return ps

def data_gen(profile, Q,R,drop_random_q_r, percent_to_drop, cfg=None):
    voltages = []
    timess = []
    statess=[]
    current_values=[]
    Q_,R_=[],[]
    generator = partial(profile.generate, unit_of_measure=1)
    for q,r in (list(itertools.product(Q[::-1], R[::-1]))):
        # Re-initialize battery model with grid parameters
        batt = BatteryElectroChemEOD()
        if drop_random_q_r:
            temp_sample= np.random.rand(1)
            if temp_sample > percent_to_drop:
                continue        

        batt.parameters['qMobile'] = q
        batt.parameters['Ro'] = r
        first_output = {"t": 18.95, "v": 4.183}
        sim_config = {"save_freq": 2}
        batt.parameters=update(deepcopy(batt.parameters))
        times, inputs, _, outputs, _ = batt.simulate_to_threshold(generator, first_output, **sim_config)

        traj = np.array([o['v'] for o in outputs]) 

        if cfg.reject_trajectory_shorter_than < len(traj) < cfg.reject_trajectory_longer_than: # Those rejected might be no well-defined trajectories
            
            current_values.append(profile) # Current values is a list of all profile objs
            # Drop voltages below 3.15 V
            times = np.array(times) #[traj>3]
            timess.append(times.astype(np.float32))
            
            voltages.append(traj.astype(np.float32))
            Q_.append(q)
            R_.append(r)
    return (voltages, current_values, timess, Q_,R_)


def init_cfg(cfg):
    """
    Sanity check for the configuration file
    """    
    if cfg.current.current_type == "variable_currents":
        assert cfg.current.N_profiles>1 # N_profiles is the maximum number of piecewise linear functions within each profile
    else:
        assert cfg.current.N_profiles==1 # This is a constant current profile
    
@hydra.main(config_path="../config/", config_name='generate_dataset')
def main(cfg):
    """
    Simple script that generates the dataset V2
    """
    init_cfg(cfg)    
    if cfg.generate_training_dataset:
        # Define the grid of parameters from which to sample (for training)
        Q = np.linspace(cfg.Q.min,cfg.Q.max,cfg.Q.res)
        R = np.linspace(cfg.R.min,cfg.R.max,cfg.R.res)


        profiles = []
        for x in range(cfg.N_currents):
            profile = CurrentProfile(cfg.current.current_type,cfg.current.min, cfg.current.max, cfg.current.N_profiles)
            profiles.append(profile)

        Q_R_tuple  = (Q,R)
        sampling_tuple = (cfg.q_r_linspace, cfg.perc_linspace)
        args_tuple = [(corr,) + Q_R_tuple + sampling_tuple for corr in profiles]
        
        outp=Parallel(n_jobs=cfg.njobs,verbose=11)(delayed(data_gen)(*(args_tuple[i] + (cfg,)))  for i in tqdm(range(len(args_tuple))))

        # Variables to be saved
        voltages = []
        timess = []
        current_values=[]
        Q_,R_=[],[]
        
        while outp:
            curr = outp.pop()
            voltages+=curr[0]
            current_values+=curr[1]
            timess+=curr[2]
            Q_+=curr[3]
            R_+=curr[4]

        total_traj = len(voltages)
        train_times, train_voltages, train_currents,Qs,Rs = timess, voltages, current_values,Q_,R_


        Path("data").mkdir(parents=True, exist_ok=True)
        print("Saving")

        
        for i in range(0,len(train_times),cfg.chunk_size):
            with open(f"data/train_times_{i}.pkl", "wb") as fp:
                pickle.dump(train_times[i:i+cfg.chunk_size], fp)
            with open(f"data/train_voltages_{i}.pkl", "wb") as fp:
                pickle.dump(train_voltages[i:i+cfg.chunk_size], fp)
            with open(f"data/train_currents_{i}.pkl", "wb") as fp:
                pickle.dump(train_currents[i:i+cfg.chunk_size], fp)
            with open(f"data/train_Qs_{i}.pkl", "wb") as fp:
                pickle.dump(Qs[i:i+cfg.chunk_size], fp)
            with open(f"data/train_Rs_{i}.pkl", "wb") as fp:
                pickle.dump(Rs[i:i+cfg.chunk_size], fp)
        metadata = {"train_times": len(train_times), "chunk_size": cfg.chunk_size}  

        # Save as a json file the total number of trajectories
        with open("data/metadata.json", "w") as fp:
            json.dump(metadata, fp)
        
    else:
        print("Warning: No Training Set generated")
    
    if cfg.test_sets.generate_test_dataset:

        drop_param = 0.9
        # Compute grid resolution given the number of samples
        res = int(np.sqrt(cfg.test_sets.number_of_test*(1-drop_param)*100))
        #current_max = cfg.current.max * cfg.extrapolation_max
        Q_min = cfg.Q.min / cfg.test_sets.extrapolation_max 
        Q_max = cfg.Q.max * cfg.test_sets.extrapolation_max
        R_min = cfg.R.min / cfg.test_sets.extrapolation_max
        R_max = cfg.R.max * cfg.test_sets.extrapolation_max
        

        if cfg.test_sets.reject_training_samples:
            # We reject samples that are in the same grid as the training samples
 
            Q_left_to_reject_segment = cfg.Q.min/cfg.test_sets.rejest_training_sample_percent
            Q_right_to_reject_segment = cfg.Q.max*cfg.test_sets.rejest_training_sample_percent

            R_left_to_reject_segment = cfg.R.min/cfg.test_sets.rejest_training_sample_percent
            R_right_to_reject_segment = cfg.R.max*cfg.test_sets.rejest_training_sample_percent

            Q_left_linspace = np.linspace(Q_min,Q_left_to_reject_segment,res//2)
            Q_right_linspace = np.linspace(Q_right_to_reject_segment,Q_max,res//2)
            Q = np.concatenate((Q_left_linspace,Q_right_linspace),axis=0)

            R_left_linspace = np.linspace(R_min,R_left_to_reject_segment,res//2)
            R_right_linspace = np.linspace(R_right_to_reject_segment,R_max,res//2)
            R = np.concatenate((R_left_linspace,R_right_linspace),axis=0)

            
        else:
            
            Q = np.linspace(Q_min,Q_max,res)
            R = np.linspace(R_min,R_max,res)


        if cfg.test_sets.current_linspace and cfg.test_sets.prebaked_currents:
            raise ValueError("Cannot use both current_linspace and prebaked_currents")

        if cfg.test_sets.current_linspace:
            current_values_iter = np.linspace(cfg.current.min, cfg.current.max,cfg.test_sets.N_currents)
            num_curves = cfg.test_sets.N_currents
        elif cfg.test_sets.prebaked_currents:
            path = hydra.utils.to_absolute_path("data/variable_test_currents.json")
            with open(path, "r") as fp:
                self_profiles, current_profiles = json.load(fp)
            num_curves = len(current_profiles)
        else:
            raise KeyError("No current_linspace or prebaked_currents")

        profiles = []

        
        for x in range(num_curves):
            if cfg.test_sets.current_linspace:
                profile = CurrentProfile(cfg.current.current_type,cfg.current.min, cfg.current.max, cfg.current.N_profiles, val=current_values_iter[x])
            elif cfg.test_sets.prebaked_currents:
                profile = CurrentProfile(cfg.current.current_type,cfg.current.min, cfg.current.max, 
                                        cfg.current.N_profiles, self_profiles=self_profiles[x], current_profiles=current_profiles[x])
            else:
                profile = CurrentProfile(cfg.current.current_type,cfg.current.min, cfg.current.max, cfg.current.N_profiles)
            profiles.append(profile)

        Q_R_tuple  = (Q,R)
        sampling_tuple = (True, drop_param) # We drop 90% of Q and R
        args_tuple = [(corr,) + Q_R_tuple + sampling_tuple for corr in profiles]
        
        outp=Parallel(n_jobs=cfg.njobs,verbose=11)(delayed(data_gen)(*(args_tuple[i] + (cfg,)))  for i in tqdm(range(len(args_tuple))))

        # Variables to be saved
        voltages = []
        timess = []
        current_values=[]
        Q_,R_=[],[]
        for i in range(len(outp)):
            voltages+=outp[i][0]
            current_values+=outp[i][1]
            timess+=outp[i][2]
            Q_+=outp[i][3]
            R_+=outp[i][4]

        # Generating Test Set
        total_traj = len(voltages)
        print(f"Total number of test QxR {len(Q)*len(R)}")
        print(f"Total number of test currents {len(set(current_values))}")
        print(f"Total Test Trajectory {total_traj}")
        rejected = len(Q)*len(R)*len(set(current_values)) - len(voltages)
        print(f"Test Trajectory Rejected {rejected}")
        test_set = np.arange(total_traj)

        # Generating Test Set
        test_times, test_voltages, test_currents, Qs,Rs = return_entries(test_set, timess, voltages, current_values,Q_,R_)
    
        Path("data").mkdir(parents=True, exist_ok=True)
        for i in range(0,len(test_times),cfg.chunk_size):
            with open(f"data/test_times_{i}.pkl", "wb") as fp:
                pickle.dump(test_times[i:i+cfg.chunk_size], fp)
            with open(f"data/test_voltages_{i}.pkl", "wb") as fp:
                pickle.dump(test_voltages[i:i+cfg.chunk_size], fp)
            with open(f"data/test_currentss_{i}.pkl", "wb") as fp:
                pickle.dump(test_currents[i:i+cfg.chunk_size], fp)
            with open(f"data/test_Qs_{i}.pkl", "wb") as fp:
                pickle.dump(Qs[i:i+cfg.chunk_size], fp)
            with open(f"data/test_Rs_{i}.pkl", "wb") as fp:
                pickle.dump(Rs[i:i+cfg.chunk_size], fp)
        metadata = {"test_times": len(test_times), "chunk_size": cfg.chunk_size}   

        # Save as a json file the total number of trajectories
        with open("data/test_metadata.json", "w") as fp:
            json.dump(metadata, fp)

    else:
        print("Warning: No Test Set generated")
    

if __name__=="__main__":
    main()