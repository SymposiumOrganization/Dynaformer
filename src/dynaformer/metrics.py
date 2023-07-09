import numpy as np
import scipy.stats as stats

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
    
    
