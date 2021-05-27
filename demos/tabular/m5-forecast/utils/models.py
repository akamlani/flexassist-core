import numpy as np 
from   typing import List, Tuple 

import tensorflow as tf 
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models 

from flexassist.integrations.tensorflow.modeler import ModelTemplate


class ModelForecastDense(ModelTemplate):
    def __init__(self, input_shape:Tuple[int, int], num_timesteps:int, output_width:int, num_features:int, horizon:int, **kwargs):
        # initialize base model impelemntation
        super().__init__(input_shape, num_features, **kwargs)
        self.num_timesteps = num_timesteps
        self.output_width  = output_width 
        self.num_features  = num_features
        self.horizon       = horizon  

        self.dense1 = layers.Dense(units=256, activation='relu', name='dense_1', input_shape=input_shape)
        self.dense2 = layers.Dense(units=128, activation='relu', name='dense_2')
        self.dense3 = layers.Dense(units=64,  activation='relu', name='dense_3')
        self.dense4 = layers.Dense(units=num_features, name='output')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)


class ModelForecastLSTM(ModelTemplate):
    def __init__(self, input_shape:Tuple[int, int], num_timesteps:int, output_width:int, num_features:int, horizon:int, **kwargs):
        # initialize base model impelemntation
        super().__init__(input_shape, num_features, **kwargs)
        self.num_timesteps = num_timesteps
        self.output_width  = output_width 
        self.num_features  = num_features 
        self.horizon       = horizon 

        # define input shape: (n_steps, n_features)
        # input as 3D: [samples][steps][features]

        # intialize core layers
        # input_shape: STMs should be limited to 200-400 timesteps
        self.lstm1 = layers.LSTM(256, activation='relu', return_sequences=True,  name='lstm_1', input_shape=input_shape)
        self.lstm2 = layers.LSTM(128, activation='relu', return_sequences=False, name='lstm_2')
        # intialize dense layers 
        self.dense_block_layer  = self.dense_block([(256, 0.1), (128, 0.1), (64, 0.1)])
        # initalize output block
        self.output_layer       = self.output_block(num_features, output_type='regressor')
        # Shape: (outputs) => (1, outputs)
        # tf.keras.layers.Reshape([-1, self.num_features]) 

    def call(self, inputs, training=False):
        # send input through forward direction 
        # Shape [batch, time, features] => [batch, time, lstm_units]
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        # head 
        # Shape => [batch, time, features]
        for layer in self.dense_block_layer:
            x = layer(x, training=training) if isinstance(layer, layers.Dropout) else layer(x) 
        # forward pass through output layer            
        return self.output_layer(x)


class ModelForecastLSTMOneShot(ModelTemplate):
    def __init__(self, input_shape:Tuple[int, int], num_timesteps:int, output_width:int, num_features:int, horizon:int, **kwargs):
        # initialize base model impelemntation
        super().__init__(input_shape, num_features, **kwargs)
        self.num_timesteps = num_timesteps
        self.output_width  = output_width 
        self.num_features  = num_features 
        self.horizon       = horizon 

        # define input shape: (n_steps, n_features)
        # input as 3D: [samples][steps][features]
        # intialize core layers
        # input_shape: STMs should be limited to 200-400 timesteps
        self.lstm1 = layers.LSTM(256, activation='relu', return_sequences=True,  name='lstm_1', input_shape=input_shape)
        self.lstm2 = layers.LSTM(128, activation='relu', return_sequences=False, name='lstm_2')
        # intialize dense layers
        # Shape => [batch, out_steps*features] 
        dense_info = [(num_units*output_width, 0.2)for num_units in [256, 128, 64]]
        self.dense_block_layer  = self.dense_block(dense_info)
        # initalize output block
        self.output_layer       = self.output_block(num_features*output_width, output_type='regressor')
        # Shape => [batch, out_steps, features]
        self.reshape_layer      = layers.Reshape([output_width, num_features])

    def call(self, inputs, training=False):
        # send input through forward direction 
        # Shape [batch, time, features] => [batch, time, lstm_units]
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        # head 
        # Shape => [batch, time, features]
        for layer in self.dense_block_layer:
            x = layer(x, training=training) if isinstance(layer, layers.Dropout) else layer(x) 
        # forward pass through output layer            
        x = self.output_layer(x)
        return self.reshape_layer(x)



class ModelForecastBiLSTM(ModelTemplate):
    def __init__(self, input_shape:Tuple[int, int], num_timesteps:int, output_width:int, num_features:int, horizon:int, **kwargs):
        # initialize base model impelemntation
        super().__init__(input_shape, num_features, **kwargs)
        self.num_timesteps = num_timesteps
        self.output_width  = output_width 
        self.num_features  = num_features 
        self.horizon       = horizon 

        # intialize core layers
        self.bilstm1 = layers.Bidirectional(
            layers.LSTM(50, activation='relu', return_sequences=True, name='lstm_1'), 
            input_shape=input_shape,
            name = 'bilstm_1'
        )
        self.bilstm2 = layers.Bidirectional(
            layers.LSTM(50, activation='relu', return_sequences=False, name='lstm_2'),
            name = 'bilstm_2'
        )
        # intialize dense layers 
        self.dense_block_layer  = self.dense_block([(256, 0.1), (128, 0.1), (64, 0.1)])
        # initalize output block
        self.output_layer       = self.output_block(num_features, output_type='regressor')

    def call(self, inputs, training=False):
        # send input through forward direction 
        x = self.bilstm1(inputs)
        x = self.bilstm2(x)
        # head 
        for layer in self.dense_block_layer:
            x = layer(x, training=training) if isinstance(layer, layers.Dropout) else layer(x) 
        # forward pass through output layer            
        return self.output_layer(x)
