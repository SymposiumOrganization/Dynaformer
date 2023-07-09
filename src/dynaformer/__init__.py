from . import data_modules
from . import models
from . import metrics
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np 

class CurrentProfile:
    """
    Class that returns current generator object which method generate() can be passed into the battery simulation model
    """
    def __init__(self, type, min_val, max_val, max_num_profiles,val=None, self_profiles=None, current_profiles=None):
        self.min_val = min_val
        self.max_val = max_val
        self.max_num_profiles = max_num_profiles
        if type == "constant_currents":
            # Generate current between two values min_val and max_val
            if val==None:
                self.val = self.get_scalar_current(self.min_val,self.max_val)
            else:
                self.val = val
            self.current_profiles = [self.val]
            self.times_profile = []
        elif type == "variable_currents":
            n_profiles = random.randint(1,self.max_num_profiles)
            val = 3000
            if self_profiles is None:
                self.times_profile = self.get_random_integer_partion(val, n_profiles)
            else:
                self.times_profile = self_profiles

            if current_profiles is None:
                self.current_profiles = [self.get_scalar_current(self.min_val,self.max_val) for i in range(n_profiles + 1)]
            else:
                self.current_profiles = current_profiles    
        else:
            raise ValueError("Invalid type, must be either variable_currents or constant_currents")
            # Gene

    def generate(self,t,x = None, unit_of_measure=1):
        """
        Given a time, return the current
        args:
            cumulative_step_size: current time [step_size]
            unit_of_measure: the unit of measure of the current [s]
        """
        if self.times_profile == []:
            return {"i":  self.current_profiles[0]} # We are in the constant loading case
        
        for i,t_profile in enumerate(self.times_profile):
            if t*unit_of_measure < t_profile:
                return {"i": self.current_profiles[i]}
        return {"i": self.current_profiles[-1]}

    def get_current_info(self):
        """
        Return the current profiles and times
        """
        return self.current_profiles, self.times_profile
    
    def return_current_profile_array(self, time_denominator=1, pad_profiles=False):
        """
        DEPRECATED (Used for deepOnet)
        Return the current profiles and times as a single list with the current profiles first and then the times.
        The total dimension of the array is the number of current profiles times 2 (current and time) minus 1 (the last time)
        args:
            time_denominator: the time is divided by this value to get the time to values similar to the current profile 
            pad_profiles: if the total profiles are shorter that the max number of profiles, pad the profiles with 0
        """
        res = []
        res.extend(self.current_profiles)
        res.extend([x/time_denominator for x in self.times_profile])
        assert len(res) == len(self.current_profiles) * 2 -1
        if pad_profiles and self.times_profile == []:
            pass
            #print('Warning. Pad-profile ignored')
        if pad_profiles and self.times_profile:
            while len(res) < self.max_num_profiles * 2  + 1:
                res.append(0)
        return res

    @staticmethod
    def get_scalar_current(min,max):
        """
        Get a random current for an interval between min and max
        """
        val = (max-min)*random.random() + min
        return val

    def get_current_profile(self, max_length, step_size = 2):
        """
        From a current profile, get the current time series
        args:
            max_length: the maximum length of the time series
            step_size: the step size of the time series [s]
        """
        current_profile = []
        for t in np.arange(0,max_length,step_size):
            value = self.generate(t,x=None)["i"]
            current_profile.append(value)
        return current_profile
    
    @staticmethod
    def get_random_integer_partion(val, number_of_partitions):
        """
        Get a random integer partition.
            args:
                val: Maximum value of the partition
                number_of_partitions: number of partitions
            returns:
                partition: a list of integers
        """
        # Get the range of the partition
        partition_range = range(val)
        # Get the number of partitions
        n_partitions = number_of_partitions
        # Get the partition
        partition = random.sample(partition_range,n_partitions)
        return sorted(partition)
    



def get_scalar(measurement):
    """
    Given a measurement, returns the associated minmax scalar
    """
    scalar = MinMaxScaler()
    scalar.fit(measurement)
    return scalar

def compute_scalers(data_dir, is_val):
    """
    Compute the scalar based on the training data
    """
    if not is_val:
        x= np.load(data_dir / 'training_states_and_observable.pkl', allow_pickle=True)
        y= np.load(data_dir / 'training_voltages.pkl',  allow_pickle=True)
        t= np.load(data_dir / 'training_times.pkl',  allow_pickle=True)
        i_c= np.load(data_dir / 'training_currents.pkl',  allow_pickle=True)
    else:
        x= np.load(data_dir / 'test_states_and_observable.pkl', allow_pickle=True)
        y= np.load(data_dir / 'test_voltages.pkl',  allow_pickle=True)
        t= np.load(data_dir / 'test_times.pkl',  allow_pickle=True)
        i_c= np.load(data_dir / 'test_currents.pkl',  allow_pickle=True)
    
    if x.shape[-1] == 8:
        x_scaler = get_scalar(np.reshape(x,(-1,8)))
    elif x.shape[-1] == 9:
        x_scaler = get_scalar(np.reshape(x,(-1,9)))
    else:
        raise ValueError("Invalid shape")
    t_scaler = get_scalar(t)
    y_scaler = get_scalar(y)

    try: 
        i_scaler = get_scalar(i_c)
    except:
        i_c =np.array([np.array(x[0].return_current_profile_array(time_denominator=1000,pad_profiles=True)) for x in i_c])
        i_scaler = get_scalar(i_c)
    
    return x_scaler,t_scaler,y_scaler,i_scaler