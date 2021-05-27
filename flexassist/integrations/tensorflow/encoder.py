import  numpy  as np 
import  pandas as pd 
from    typing      import List 
from    dataclasses import dataclass

import  tensorflow.keras.layers as layers 
from    tensorflow import feature_column as fc

@dataclass
class EntityProperty:
    name:str                  # name of categorical embedding, e.g. col
    in_dim:int                # cardinality of input dimension
    emb_dim:int     = None 
    in_shape:int    = 1 


class TFCatEncoder(object):
    "Entity Embedding Categorical Encoder"
    def __init__(self, properties:List[EntityProperty], **kwargs):        
        super().__init__(**kwargs)
        # create a series of entity embedding blocks
        self.entity_block = { 
            prop.col: self._encoder(prop.col,  prop.in_dim, prop.emb_dim)
            for prop in properties
        }

    def _encoder(self, col:str, in_dim:int, emb_dim:int=None, spatial_dropout:float=0.2) -> dict:
        emb_dim = TFCatEncoder.get_emb_size(in_dim) if not emb_dim else emb_dim
        emb     = layers.Embedding(in_dim+1, emb_dim, name=f'{col}_emb')  #(inp)
        sdrop   = layers.SpatialDropout(spatial_dropout)                  #(emb)
        emb_vec = layers.Flatten(name=f'flattened_{col}_emb')             #(sdrop)
        return {'emb': emb,  'reg': sdrop, 'emb_vec': emb_vec}

    @classmethod
    def _encoder_functional(cls, name:str, in_shape:int, in_dim:int, emb_dim:int, **kwargs) -> tuple:
        return cls.entity_emb_functional(name, in_shape, in_dim, emb_dim) 

    @classmethod
    def get_emb_size(cls, cardinality:int) -> int:
        # embedding_dim = max(min((cardinality)//2, 50),2)
        # embedding_dim = int(sqrt(cardinality))
        return int( min( np.ceil(cardinality // 2), 50) )

    @classmethod
    def shared_emb(cls, inputs:List[layers.Input], emb_dim:int) -> dict:
        emb_layer = layers.Embedding(emb_dim)
        return {inp: emb_layer(inp) for inp in inputs}

    @classmethod
    def entity_emb_functional(cls, name:str, in_shape:int, in_dim:int, emb_dim:int=None) -> tuple:
        "create a shallow entity embedding with integer ids for sparse interactions"
        # for single columnar, entity embeddings should have input shape of (1)
        # Embedding(traiable=False) to not be updated during training
        # Embedding(weights=embedding_dict[col]) to preinitalize with weights
        emb_dim = TFCatEncoder.get_emb_size(in_dim) if not emb_dim else emb_dim
        inp     = layers.Input(shape=(in_shape,), name=name)
        outp    = layers.Embedding(in_dim+1, emb_dim, name=f'{name}_emb')(inp)
        emb_vec = layers.Flatten(name=f'flattened_{name}_emb')(outp)
        return inp, outp, emb_vec



    # def call(self, inputs, training=False):
    #     # send input through forward direction 
    #     n_inputs = len(inputs)
    #     # inputs: forward through embedding and flatten layers
    #     outputs  = []
    #     for k,v in inputs.items():
    #         entity_block = self.entity_block.get(k)
    #         x = entity_block["emb"](v)
    #         x = entity_block['reg'](x)
    #         x = entity_block['emb_vec'](x)
    #         outputs.append(x)
    #     enc_concat = layers.concatenate(outputs, axis=1, name='concat')  \
    #                  if n_inputs > 1 else outputs[-1]
    #     # forward pass through FC dense block (dense, dropout)
    #     x = enc_concat
    #     for layer in self.dense_block:
    #         x = layer(x, training=training) if isinstance(layer, layers.Dropout) else layer(x) 
    #     # forward pass through output layer            
    #     return self.output_layer(x)



