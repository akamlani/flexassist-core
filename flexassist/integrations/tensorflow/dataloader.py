import  numpy  as np 
import  pandas as pd 
from    typing import List, Tuple 

import  tensorflow as tf 
import  tensorflow.keras as keras 

gen_dataset       = lambda data:    tf.data.Dataset.from_tensor_slices(data)
get_dataset_shape = lambda dataset: tf.shape(list(dataset.as_numpy_iterator()))
get_dataset_spec  = lambda dataset: dataset.element_spec
get_example       = lambda dataset: next(iter(dataset))       

#print('Input shape:',  wdl.example[0].shape)
#print('Output shape:', model(wdl.example[0]).shape)
forward_pass      = lambda model, x_in: model(x_in)

class WindowDataLoader(object):
    def __init__(self, input_width:int=14, label_width:int=1, horizon:int=28):
        self.input_width         = input_width             # prior time steps (lags)
        self.label_width         = label_width             # width of time steps output to provide 
        self.horizon             = horizon                 # equivalent to shift or offset
        self.total_window_size   = input_width + horizon   # input window and horizon 
        self.input_slice         = slice(0, input_width)
        self.input_indices       = np.arange(self.total_window_size)[self.input_slice]

        # these labels are in regards to the horiozn 
        self.label_start         = self.total_window_size - label_width
        self.labels_slice        = slice(self.label_start, None)
        self.label_indices       = np.arange(self.total_window_size)[self.labels_slice]
        
    def register_frame(self, df:pd.DataFrame, columns:List[str]) -> None:
        "register training frame"
        # these labels are in regards to the dataframe input
        self.registered_columns  = columns
        self.column_lookup       = {col: df.columns.get_loc(col) for col in columns if col in df}
        self.column_indices      = list(self.column_lookup)
        
    def register_metadata(self, meta_data:dict) -> None:
        "register additional configuration used for training"
        self.meta_config = meta_data

    def split_window(self, features):
        """handles both single-output and multi-output
        
        Example:
        example_window = tf.stack([
            np.array(train[:wc.total_window_size]),
            np.array(train[100:100+wc.total_window_size]),
            np.array(train[200:200+wc.total_window_size])]
        )
        example_inputs, example_labels = wc.split_window(example_window)
        .example_info(example_window, example_inputs, example_labels)
        """
        inputs = features[:, self.input_slice,  :]
        labels = features[:, self.labels_slice, :]
        # slicing doesn't preserve static shape information, so set the shapes
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels
        
    def make_dataset(self, data:np.array, shuffle_en:bool=True, batch_size:int=32, stride:int=1):
        "make a tensorflow dataset"
        data = np.array(data, dtype=np.float32)
        ds   = tf.keras.preprocessing.timeseries_dataset_from_array (
                  data            = data,
                  targets         = None,
                  sequence_length = self.total_window_size,
                  sequence_stride = stride,
                  shuffle         = shuffle_en,
                  batch_size      = batch_size
        )
        ds = ds.map(self.split_window, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds    
    
    def example_window_info(self, window:np.array, inputs:np.array, labels:np.array):
        return '\n'.join([
            f'Shapes: (batch, time, features)',
            f'Window shape: {window.shape}',
            f'Inputs shape: {inputs.shape}',
            f'Labels shape: {labels.shape}'])    
    
    def example_info(self, inputs:np.array, labels:np.array):
        return '\n'.join([
            f'Shapes: (batch, time, features)',
            f'Inputs shape: {inputs.shape}',
            f'Labels shape: {labels.shape}'])        

    def __repr__(self):
        return '\n'.join([
            f'Total window size:           {self.total_window_size}',
            f'Input indices:               {self.input_indices}',
            f'Label indices:               {self.label_indices}',
            f'Registered column name(s):   {self.registered_columns}',
            f'Registered column indice(s): {self.column_indices}'
        ])
