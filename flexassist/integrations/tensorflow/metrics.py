import numpy as np 
import tensorflow as tf 
import tensorflow.keras as tf 
import tensorflow.keras.metrics as metrics 

# aggregate Metrics
# metrics across each particular horizon day

mae_base = lambda y, yhat: np.abs(yhat - y).mean()
mae_tf   = lambda y, yhat: metrics.mean_absolute_error(y, yhat).numpy()
# RMSE: punishing of forecasting errors



def custom_smape(x, x_):
    import tensorflow.keras.backend as K
    return K.mean(2*K.abs(x-x_)/(K.abs(x)+K.abs(x_)))

def mape(true, predicted):        
    inside_sum = np.abs(predicted - true) / true
    return round(100 * np.sum(inside_sum ) / inside_sum.size,2)
