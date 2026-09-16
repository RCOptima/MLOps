from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from trainers.transformer import Transformer

 
# def register_trainer(name: str):
#     def wrapper(cls: type[BaseTrainer]):
#         TRAINER_REGISTRY[name] = cls
#         return cls
#     return wrapper

class BaseTrainer(ABC):
    """
    Every dataset-specific trainer subclasses this and implements
    build_pipeline(). Everything else (Optuna study, nested MLflow
    runs, metric computation, transform monitoring, registration) is
    handled once, here, for all of them.
    """
 
    name: str
    primary_metric: str = "rmse"
    direction: str = "minimize"  # "maximize" for accuracy / f1 / r2
 
    @abstractmethod
    def build_pipeline(self, trial: "optuna.Trial") -> Pipeline:
        """Return an unfitted sklearn Pipeline for one Optuna trial:
        Pipeline([("transform", <your BaseTransform>), ("model", <your estimator>)]).
        Call trial.suggest_int / suggest_float / suggest_categorical inline
        for anything you want Optuna to tune -- Optuna records whatever you
        sample as this trial's params, so there's no separate hyperparams
        dict to maintain."""
        raise NotImplementedError
 
 
# example
class ChurnTrainer(BaseTrainer):
    name = "churn"
    primary_metric = "rmse"
    direction = "minimize"
 
    def build_pipeline(self, trial: "optuna.Trial") -> Pipeline:
        from sklearn.ensemble import RandomForestRegressor
 
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            max_depth=trial.suggest_categorical("max_depth", [5, 10, 20, None]),
            random_state=42,
        )
        return Pipeline(steps=[("transform", Transformer()), ("model", model)])
 


TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {
    "churn": ChurnTrainer,
    # a new project adds one line here
}