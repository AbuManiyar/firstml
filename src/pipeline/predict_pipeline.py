import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object

@dataclass
class InputData:
    gender: str = None
    race_ethnicity : str = None
    parental_level_of_education : str = None
    lunch : str = None
    test_preparation_course : str = None
    reading_score: int = None
    writing_score: int = None
    
class Transformed():
    def __init__(self):
        self.pre_data = InputData()
        
    def transform(input_data):
        
        pass
        #self.pre_data = InputData(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)