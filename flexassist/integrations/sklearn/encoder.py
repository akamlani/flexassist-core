import  enum 
import  numpy  as np 
import  pandas as pd 

from    sklearn import preprocessing as proc 
from    typing  import List


class EncoderType(enum.Enum):
    OneHot      = 1
    Ordinal     = 2
    Binary      = 3
    Label       = 4
    Cyclic      = 5 

class CatEncoder(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

    @classmethod
    def lookup_encoder(cls, encoder_type:EncoderType):
        encoder_types = {
            EncoderType.Label:  proc.LabelEncoder, 
            EncoderType.Binary: proc.LabelBinarizer,
            EncoderType.OneHot: proc.OneHotEncoder    
        }
        return encoder_types.get(encoder_type, EncoderType.Label)

    def encode(self, df:pd.DataFrame, cat_features:List[str], encoder_type:EncoderType=EncoderType.Label) -> tuple:
        "applicable for OHE for the matrix, where we can encode entire list of features"
        encoder   = CatEncoder.lookup_encoder(encoder_type)()
        encoder.fit(df[cat_features])
        df_cat_tr = encoder.transform(df[cat_features].values)
        #encoder.classes_
        return (encoder,  df_cat_tr)

