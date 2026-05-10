import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,

)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models={
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRegressor":XGBRegressor()
            }
            
            params = {
    "LinearRegression": {
        "fit_intercept": [True, False],
        "positive": [True, False]
    },
    "DecisionTree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["squared_error", "friedman_mse"]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["auto", "sqrt", "log2"]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
        "loss": ["linear", "square", "exponential"]
    },
    "KNeighbors": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # Manhattan vs Euclidean
    },
    "CatBoostRegressor": {
        "iterations": [200, 500],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128]
    },
    "XGBRegressor": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 5, 10]
    }
}

            
                        
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            return r2
        except Exception as e:
            raise CustomException(e,sys)