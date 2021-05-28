import  mlflow
import  mlflow.pytorch
from    mlflow import log_metric, log_param, log_artifact
from    mlflow.models import Model  #Model.load
from    mlflow import pyfunc        #mlflow.pyfunc.load_model


track_param    = lambda k,v: mlflow.log_param(k,v)
track_metric   = lambda k,v: mlflow.log_metric(k,v) 
track_artifact = lambda x:   mlflow.log_artifact(x)
# autologging for tf.keras is handled by mlflow.tensorflow.autolog(), not mlflow.keras.autolog().
track_auto_tf  = lambda tf:  mlflow.tensorflow.autolog() if tf else mlflow.keras.autolog()
track_auto_skl = lambda:     mlflow.sklearn.autolog()  


class Tracker(object):
    """MLFlow Template Wrapper Class"""
    def __init__(self, name, project_name, experiment_dir="dev-environment/experiments", tags=None):
        super().__init__(name, project_name, experiment_dir, tags)

        # default configurations
        self.experiment_name = project_name
        self.server_uri, self.port_num = ("localhost", 5000)
        # configuration mlflow
        mlflow.set_tracking_uri(f"http://{self.server_uri}:{self.port_num}")
        mlflow.set_experiment(f"{self.experiment_name}")

    def log_parameters(self, params_dict):
        """log a set of hyperparameters
        >>> with mlflow.start_run():
                mlflow.log_param("num_dimensions", 8)
                mlflow.log_param("tuning", tuning_params_dict)
        """
        for idx, (name, value) in enumerate(params_dict.items()):
            track_param(name, value)

    def log_metric(self, metrics_dict):
        """log a set of evaluation metrics
           stepwise(e.g. epoch, training iteration) not required to be successive vs timestep

        >>> learning rate, training loss, validation loss
        >>> with mlflow.start_run():
                mlflow.log_metric(key="accuracy", value=0.1)
                mlflow.log_metric(key="quality",  value=2*epoch, step=epoch)
        """
        for idx, (name, value) in enumerate(metrics_dict.items()):
            track_metric(name, value)

    def log_artifact(self, artifact):
        """track resource artifact

        >>> with mlflow.start_run():
                mlflow.log_artifact(checkpoint_path)
                mlflow.log_artifact(filename.txt)
        """
        track_artifact(artifact)

    def log_model_artifact(self, model, model_name="model"):
        """log trained model artifact
        
        >>> with mlflow.start_run():
                log_model_artifact(model.pkl)
        """
        # Log the Model, tracking to respective project
        mlflow.sklearn.log_model(model, model.__class__.__name__)
