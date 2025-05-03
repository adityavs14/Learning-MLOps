import mlflow
from mlflow.models import infer_signature
import mlflow.pytorch

# set url
class logger():
    
    def __init__(self, experiment_name):
        mlflow.set_tracking_uri(uri = "http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name=experiment_name)
        # mlflow.start_run()
        
    def start_run(self,):
        mlflow.start_run()
        
    def end_run(self,):
        mlflow.end_run()
        
    def log_param(self,params):
        mlflow.log_params(params)
            
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)
        
    def create_artifacts(self, signature, model):
        
        model_info = mlflow.pytorch.log_model(model, "model",
                                              signature=signature)