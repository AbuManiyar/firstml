import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            nm_col = ['writing_score', 'reading_score']
            cat_col = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                       'test_preparation_course']
            
            num_pipeline = Pipeline(
                [
                 ('imputer', SimpleImputer(strategy='median')),
                 ('scaler', StandardScaler(with_mean=False))   
                ]
            )            
            
            cat_pipeline = Pipeline(
                [
                 ('imputer', SimpleImputer(strategy='most_frequent')),
                 ('onehotencoder', OneHotEncoder()),
                 ('Scaling', StandardScaler(with_mean=False))   
                ]
            )
            
            preprocessor = ColumnTransformer([
                ('num',num_pipeline , nm_col ),
                ('cat', cat_pipeline, cat_col)
            ]
            )
            
            logging.info('columns scaling completed')
            logging.info('Categorical columns encoding competed')
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
          
          
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj)   

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
#Code written for testing            
#'''            
if __name__ == '__main__':
    
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion() 
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_training = ModelTrainer()
    model_training.initiate_model_trainer(train_arr,test_arr)
  #'''    