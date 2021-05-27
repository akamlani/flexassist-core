import numpy as np 
from   typing import List, Tuple 

import tensorflow as tf 
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models 

class ModelTemplate(tf.keras.Model):
    def get_trainable_weights(model):           return model.trainable_weights
    def get_parameters(model):                  return model.trainable_variables
    def get_weights(model):                     return model.weights
    def get_layer_weights(layer):               return layer.weights
    # pass example through forward pass first
    def get_features(model):                    return [layer.output for layer in model.layers]
    def get_layer_features(model, layer_name):  return model.get_layer(name=layer_name).output
    def extract_features(model, features):      return models.Model(inputs=model.inputs, outputs=features)
    def emb_lookup(emb, inputs):                return tf.nn.embedding_lookup(emb, inputs)

    def __init__(self, input_shape:int, num_outputs:int, **kwargs):
        super().__init__(**kwargs)
        self.in_dim      = input_shape 
        self.num_outputs = num_outputs 

    def load_model(self, model_name:str):
        """
        Examples:
        >>> laod_model("best_model_HDF5_format.h5")             # Load model using the HDF5 format
        """
        return models.load_model(model_name)

    def save_model(self, model, model_name:str):
        """
        Examples:
        >>> save_model(model, "best_model_HDF5_format.h5")      # Save model using the HDF5 format
        """
        model.save(model_name)

    def dense_block(self, meta_config:List[Tuple[int,float]], **kwargs) -> list:
        """create a dense block of layers (dense, dropout) 

        block = .dense_block([(256,0.2),(256,0.2)])
        """
        from   functools import reduce
        import tensorflow.keras.layers as layers
        num_blocks = len(meta_config)
        # add additional batchnrom as precursor
        bn = layers.BatchNormalization(name='bn_0')
        components = [ 
            ( layers.Dense(units  = unit, activation="relu", name=f"denseb_{idx}"),
              layers.Dropout(rate = proba, name=f"dropoutb_{idx}"),
              layers.BatchNormalization(name=f"bn_{idx}") )
            for idx, (unit, proba) in enumerate(meta_config, start=1)
        ]
        return [bn] + list(reduce(lambda x,y: x+y, components))

    def output_block(self, num_outputs:int, output_type:str='regressor'):
        import tensorflow.keras.layers as layers
        # regressor 
        if output_type == 'regressor':
            self.regressor  = layers.Dense(units=num_outputs, activation='linear', name=f'{output_type}_output')
            return self.regressor
        # classifier
        else:     
            num_outputs, activation = ((num_outputs -1), 'sigmoid') if num_outputs == 2 else (num_outputs, 'softmax')
            self.classifier = layers.Dense(units=num_outputs, activation=activation, name=f'{output_type}_output')
            return self.classifier

    def build_graph(self):
        """create a new model from existing with input shape and forward pass due to subclassing 

        >>> model.build_graph().summary()
        >>> model = model.build_graph()
        """
        x = layers.Input(shape=(self.in_dim))
        return models.Model(inputs=[x], outputs=self.call(x))
