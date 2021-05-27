"""
Data Model Class implementations for the wandb runner
"""
import  numpy  as np
import  pandas as pd  
import  enum 

import  os 
from    pathlib import Path 
from    typing  import List, Callable, Optional
from    dataclasses import dataclass, asdict, field

import  wandb 

# Grouping: Pre-Processing | In-Processing | Post-Processing
class JobTrackingCycle(str, enum.Enum):
    # Pre-Processing
    Core            = 'launchpad'                    # {launch space, default space}
    Ingestion       = 'ingestion'                    # {Ingestion and Transform: ETL, ELT}
    Analysis        = 'analysis'                     # {EDA, Statistical Analysis}
    Prepare         = 'dataprep'                     # {prepare the data for }
    # In-Model Processing
    Experiment      = 'training'                     # {model experiments, training}
    Sweep           = 'tuning'                       # {e.g tuning sweep}
    # Post-Processing
    Optimization    = 'optimization'                 # {knowledge distillation, quanitzation, compiled intermediate formats}
    Inference       = 'inference'                    # {inference for predictions of new records}
    Scoring         = 'scoring'                      # {scoring on a batch dataset for held out data}
    Evaluation      = 'testing'                      # {post processing quantitative offline}
    Reporting       = 'reporting'                    # {post processing}
    Monitor         = 'monitor'                      # Not currently defined: N/A

# actions for job or artifacts (to be associated with job-type)
class ActionType(enum.Enum):
    Info            = "info"                         # equivalent to get-info
    Status          = "status"                       # equivalent to get-status
    Upload          = "upload"                       # equivalent to create-dataset
    Download        = "download"                     # equivalent to get-dataset
    Execution       = 'execution'                    # equivalent to processing

# artifact prefix generation of creation type (Store Type)
class ArtifactStorage(enum.Enum):
    Config          = "config"                       # configuration properties
    Dataset         = "datacatalogue"                # original raw dataset to be used
    Transform       = "transformers"                 # equivalent to preprocess
    Feature         = "feature"                      # equivalent to feature-vector, for feature discovery, via pushdown
    DataProduct     = "dataproduct"                  # equivalent to ready available data product for reuse 
    Experiment      = "mlproduct"                    # equivalent to model-artifact
    Evaluation      = "evaluation"                   # equivalent to evaluation-store

# type of storage request
class ArtifactWrite(enum.Enum):
    AddFile         = 1                              # e.g. artifact.add_file('model.h5')
    AddDir          = 2                              # e.g. artifact.add_dir('images')
    AddRef          = 3                              # e.g. for recording metadata, but not actual data
    AddBinary       = 4                              # e.g. figures
        


@dataclass  
class JobTrackInfo:
    "select job tracking information to populate from wandb.init"
    project:str                                             # name of the project sending the run to (else placed in 'Uncategorized' project)
    entity:str                                              # entity qualifier (username or team name sending runs to), must be created beforehand
    name:Optional[str]                                      # a short display name to identify the run
    config:Optional[dict]                                   # metadata information about a job or context (e.g. baseline, staging, production)

    # Configurable with defaults
    group:str              = field(default="exp:default")   # will be qualified by experiment
    job_type:str           = field(default="")              # (e.g. <JobTrackingCycle>:<Action>)
    notes:str              = field(default="")              # free form like commit message
    tags:List[str]         = field(default_factory=list)     
    # Defaults associated with a Job 
    resume:bool            = field(default=True)
    save_code:bool         = field(default=False) 
    sync_tensorboard:bool  = field(default=True)
    
    def __post_init__(self):
        # default name: e.g. services:core
        self.name          = ":".join(['lc-svc', self.name, 'ops', wandb.util.generate_id()])
        self.group         = ":".join([self.group, wandb.util.generate_id()])
        self.job_type      = ":".join([JobTrackingCycle.Core.value, ActionType.Execution.value])
        if not self.notes:
            self.notes     = "<INSERT JOB NOTES HERE>"

    def format_job_type(self, cycle:JobTrackingCycle, action:ActionType) -> str:
        self.job_type      = ":".join([cycle.value, action.value])
        
    def build(self, job_type:str, notes:str, tags:str) -> None:
        self.job_type      = job_type
        self.notes         = notes
        self.tags          = tags



@dataclass 
class ArtifactInfo:
    "Artifact pipeline for dataset versioning and model lineage"
    name:str                                                    # unique name to reference artifacts
    type:str                                                    # {ArtifactStorageType}
    description:Optional[str]   = field(default="")             # free text string dipslaying next to artifact version in UI
    metadata:Optional[dict]     = field(default_factory=dict)   # structured metadata associated with artifact (e.g. class distributions)

    @classmethod
    def lookup_storage_type(cls, type_request:ArtifactStorage) -> str:
        "associates to .type filed of W&B Artifact"
        return (":".join([type_request.value, 'store']) )

    @classmethod
    def lookup_write(cls, cxt, write_type:ArtifactWrite=None) -> Callable:
        "retrieve function pointer for type of write storage action to be performed"
        write_lookup = lambda cxt: {
            ArtifactWrite.AddFile:      cxt.add_file,            # e.g. artifact.add_file('model.h5')
            ArtifactWrite.AddDir:       cxt.add_dir,             # e.g. artifact.add_dir('images/', path="<from given directory destination>")
            ArtifactWrite.AddRef:       cxt.add_reference,
            ArtifactWrite.AddBinary:    cxt.add                  # e.g. artifact.add(table, "<path>/my_table")
        }
        # context should be artifact reference, not job tracking runner reference
        write_fn = write_lookup(cxt).get(write_type, ArtifactWrite.AddRef)
        return write_fn 

    def build(self, name:str, type_request:ArtifactStorage, description:str, metadata:dict=None, aliases:List[str]=None):
        "construct artifact information for the dispatch"
        self.name        = name
        # e.g. type_request:ArtifactStorageType = ArtifactStorageType.DataCatalogue,
        self.type        = ArtifactInfo.lookup_storage_type(type_request)
        self.description = description
        self.metdatada   = metadata



class LifeCycleSvc(object):
    """Service responsible for initialization context of the WanDB service with appropriate info
    
    Examples:
    >>> tracker = JobTrackInfo.build(...)
    >>> run     = RunnerSvc.execute(tracker)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # silence messages
        import logging 
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.ERROR)

    @classmethod
    def execute(cls, info:JobTrackInfo):
        "execute the remote api and returns a runner (run)"
        # wandb.init(): creates local dir to save (logs, files), then streamed asynchronously
        return wandb.init(**asdict(info))

    @classmethod
    def get_context_info(cls):
        "pull relevant information from run object"
        api      = wandb.Api()
        username = wandb.run.entity
        project  = wandb.run.project
        run_id   = wandb.run.id


class LifeCycleArtifactSvc(LifeCycleSvc):
    """Service responsible for dispatching requests to the WanDB service
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_artifact(cls, cxt, name:str, type_request:ArtifactStorage) -> None:
        """ copies friles from referenced paths to construct artifact directory
        name = 'mnist:latest'
        # A new artifact version: only logged if the files in the bucket changed
        e.g. /mount/datasets/mnist -> artifacts/mnist:v0/
        """
        artifact     = cxt.use_artifact(name, type=ArtifactInfo.lookup_storage_type(type_request))   
        artifact_dir = artifact.download()
        return artifact_dir

    @classmethod
    def get_info(cls, cxt, name:str, type_request:ArtifactStorage) -> pd.DataFrame:
        """get information about the artifact
        name = 'mnist:latest'        
        """
        # run context information
        run_name = wandb.run.name
        run_id   = wandb.run.id
        # get artifact information
        artifact = cxt.use_artifact(name, type=ArtifactInfo.lookup_storage_type(type_request))   
        return pd.DataFrame.from_dict( dict(
            project  = artifact.project,
            name     = artifact.name,
            size     = artifact.size, 
            type     = artifact.type,
            version  = artifact.version,
            alias    = artifact.aliases,
            text     = artifact.description, 
            meta     = artifact.metadata 
        ), columns=['info'], orient='index').T

    def add(self, cxt, 
            info:ArtifactInfo,
            uri_src_path:str, 
            uri_dst_path:str,
            alias:List[str]              = None, 
            write_request:ArtifactWrite  = ArtifactWrite.AddRef) -> None:
        """responsible for adding dataset or reference to a repository

            Examples:
            >>> artifact_info:ArtifactInfo = ArtifactInfo.build( 
                    name = 'sales-forecast', 
                    type_request=ArtifactStorageType.Config, 
                    description=text,
                    metadata=stats_dict, 
                    aliases=['kaggle-catalogue']
                )
            >>> svc = RunnerArtifactSvc.add(cxt=runner, 
                    artifact_info   = artifact_info,
                    write_request   = ArtifactWriteType.AddRef
                    uri_src         = Path(dataset_config['dataset']['path])/'train.csv'
                    uri_dst         = 'train.csv'
                )
        """
        # wandb.log_artifact: save output of run (weights, predictions, pipeline)
        # perform Callable function pointer (e.g. .add_reference, .add_dir, .add_file)
        uri_dst   = os.path.join(ArtifactInfo.lookup_storage_type(info.type), uri_dst_path)
        # populate internal data structure
        info_dict = asdict(info)
        info_dict['type'] = info_dict['type'].value
        artifact  = wandb.Artifact(**info_dict)                                          
        ArtifactInfo.lookup_write(artifact, write_request)(uri_src_path, uri_dst)
        # execution to write the local artifact to remote storage
        # alt: run.log_artifact -> artifact.save()
        cxt.log_artifact(artifact, aliases=alias)      


    @classmethod
    def write_s3(cls, cxt, rel_path:str, bucket_name:str, info) -> str:
        """
        data = dict(
            name           = 'cnn'
            model_artifact = 'my_model.h5'
            bucket_name    = 'my_bucket'
            rel_path       = 'models/cnn'
            type           = 'model'
            uri             = 's3://my-bucket/datasets/mnist'
        )
        """
        import boto3
        # upload file to S3 bucket
        s3_client = boto3.client('s3')
        s3_client.upload_file(info.uri_src, bucket_name, Path(rel_path)/info.uri_src)
        # destination path (uri_dst), provide to .add
        return f's3://{bucket_name}/{rel_path}'

    @classmethod
    def write_nfs(cls, cxt, data, artifact_name:str, rel_path:str, mount_path:str='/mount') -> str:
        """
        data = dict(
            name           = 'cnn'
            uri            = /mount/models/cnn/my_model.h5'
            rel_path       = 'models/cnn'
            model_artifact = 'my_model.h5' 
            type           = 'model'
            uri            = 'file:///mount/datasets/mnist/'
        )
        """
        # write asset
        path = f'{mount_path}/{rel_path}/{artifact_name}'
        with open(path) as f:
            f.write(data) 
        # nfs path, provide to .add 
        return f'file:///{path}' 









   