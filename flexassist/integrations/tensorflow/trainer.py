import numpy  as np
import pandas as pd 
import datetime 
from   typing  import List 
from   pathlib import Path 
from   dataclasses import dataclass, asdict, is_dataclass

import tensorflow as tf 
import tensorflow.keras as keras 
import tensorflow.keras.layers     as layers 
import tensorflow.keras.models     as models 
import tensorflow.keras.optimizers as opt
import tensorflow.keras.losses     as losses
import tensorflow.keras.metrics    as metrics       # Metrics
import tensorflow.keras.callbacks  as callbacks     # LambdaCallback

import wandb 
from   wandb.keras import WandbCallback


@dataclass
class TrainingConfig:
    "training parameters"
    loss:str 
    metrics:List[str]
    num_epochs:int          = 10
    batch_size:int          = 32
    learning_rate:float     = 5e-4
    patience:int            = 3         # number of epochs of which no improve in training for ES to be stopped
    validation_split:int    = 0.2
    momentum:float          = 0.99
    verbosity:int           = 0

@dataclass 
class ExperimentConfig: 
    "configuration paths for model training"
    tensorboard_dir:str
    model_dir:str
    checkpoint_dir:str 
    figures_dir:str 



class LogCallback(callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def on_epoch_end(self, epoch:int, logs:dict=None):
        from sklearn.metrics import r2_score
        keys = list(logs.keys())
        metrics_dict = dict(
            epoch = epoch
        )
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        #wandb.log({'r2_score': r2_score})



class Trainer(object):
    def __init__(self, training_config:dict, cxt=None):
        def populate(data:dict, subset:dict) -> dict:
            """used to filter and populate information to a dataclass

            Examples:
            data_dict = populate(training_config['paths'], PathConfig.__annotations__)
            PathConfig(**data_dict)
            """
            return {k:v for k, v in data.items() if k in subset}
        # populate internal data structure
        self.training_params   = TrainingConfig(**populate(training_config['training_parameters'], TrainingConfig.__annotations__ ))
        self.experiment_params = ExperimentConfig(**populate(training_config['experiment_parameters'], ExperimentConfig.__annotations__ ))
        # context runner for wandb
        self.cxt = None 

    def train(self, model_name:str, model, train, validation, **kwargs) -> tuple:
        "encapsulation function to train and save the model"
        model = self.compile(model)
        model, df_history = self.fit(model_name, model, train, validation, **kwargs)
        # save the model to directory
        model_path      = str(Path(self.experiment_params.model_dir)/model_name)
        self.save(model, model_path, format='tf')
        # save the model figure architecture 
        fig_filename  = model_name + '.png'
        fig_path      = str(Path(self.experiment_params.figures_dir)/fig_filename)
        self.save_architecture(model, filename=fig_path)
        return model, df_history

    def compile(self, model):
        "compile the architecture with the ability to extract model.suammary"
        model.compile(
            optimizer = opt.Adam(float(self.training_params.learning_rate)),   
            # extend to a dict of losses for multi-output
            loss      = self.training_params.loss,                                 
            metrics   = self.training_params.metrics
        )
        return model 

    def fit(self, model_name:str, model, train, validation, **kwargs):
        """fit and train the model with training data"""
        # instantiate callbacks
        tb_log_dir      =  self.experiment_params.tensorboard_dir
        tb_log_dir      =  str(Path(tb_log_dir)/datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer          =  tf.summary.create_file_writer(tb_log_dir)
        model_filename  = '-'.join([f'model.{model_name}', '{epoch:02d}-{val_loss:.2f}.h5'])
        model_path      = str(Path(self.experiment_params.checkpoint_dir)/model_filename)

        callbacks_lst = [
            #tolerance for convergence training over epochs
            callbacks.EarlyStopping(monitor     =   'val_loss', 
                                    mode        =   'min',
                                    patience    =   self.training_params.patience,
                                    min_delta   =   1e-3),
            # tensorboard
            callbacks.TensorBoard(log_dir=tb_log_dir),
        ]
        if kwargs.get('cb_ckpt', False):
            # only saving the best checkpoints
            callbacks_lst.append( callbacks.ModelCheckpoint(model_path, save_freq ='epoch', save_best_only=False) )

        if kwargs.get('cb_wandb', False):
            callbacks_lst.append( 
                WandbCallback(
                    monitor          = 'val_loss', 
                    training_data    = validation,
                    # when training and validation are tf.data.Dataset, can just use the length
                    validation_steps = len(validation),
                    log_weights=True, 
                    log_evaluation=True,
                    #log_gradients=True, 
                )
            )

        if kwargs.get('cb_logger', True):
            callbacks_lst.append(LogCallback())

        # assumes inputs are datasets, equivalent to the following: 
        # steps_per_epoch  = TotalTrainingSamples   / TrainingBatchSize
        # validation_steps = TotalvalidationSamples / ValidationBatchSize
        history = model.fit(train, 
                            validation_data     = validation,
                            # when training and validation are tf.data.Dataset, can just use the length
                            steps_per_epoch     = len(train),
                            validation_steps    = len(validation),
                            epochs              = self.training_params.num_epochs, 
                            batch_size          = self.training_params.batch_size,
                            callbacks           = callbacks_lst,
                            use_multiprocessing = True)

        df_history = pd.DataFrame(history.history)
        return model, df_history

    def load(self, export_path:str):
        """load an existing serialized model

        Examples:
        >>> .load(self.training_config.model_export_path)
        """
        model = models.load_model(export_path)
        return model 

    def save(self, model, export_path:str, format:str='tf'):
        """serialize a model to a given path

        ._save(model, export_path)
        """
        model.save(export_path, save_format=format)

    def save_architecture(self, model, filename:str, **kwargs):
        """save model architecture diagram to a file
        Examples
        >>> model.summary()
        >>> .save_architecture(model, filename='./exports/models/model_arch.png')
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        params  = dict(show_layer_names=True, show_shapes=True, show_dtype=True, rankdir="TB")
        # for subclassing: using model.build_graph() to return new model
        return keras.utils.plot_model(model.build_graph(), to_file=filename, **params) 

    def _evaluate(self, model, X, y, batch_size:int):
        yhat            = model.predict(X)
        loss, accuracy  = model.evaluate(X, y, batch_size=batch_size, verbose=2)
        error_rate      = round((1 - accuracy) * 100, 2)
        # (1) Global Partition Metrics (2) Per Class Distribution Metrics
        metrics = {'accuracy': accuracy, 'loss': loss, 'error_rate': error_rate}
        # separate into delayed commits
        wandb.log(metrics, commit=False)
        wandb.log()
        # run.join()