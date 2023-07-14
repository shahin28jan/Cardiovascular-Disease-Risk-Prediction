import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
        
        
class CustomData:
    def __init__(self,
                 General_Health:str,
                 Checkup:str,
                 Exercise:str,
                 Skin_Cancer:str,
                 Other_Cancer:str,
                 Depression:str,
                 Diabetes:str,
                 Arthritis:str,
                 Sex:str,
                 Age_Category:str,
                 Height:float,
                 Weight:float,
                 BMI:float,
                 Smoking_History:str,
                 Alcohol_Consumption:float,
                 Fruit_Consumption:float,
                 Green_Vegetables_Consumption:float,
                 FriedPotato_Consumption:float):
                 
        
        self.General_Health=General_Health
        self.Checkup=Checkup	
        self.Exercise=Exercise
        self.Skin_Cancer=Skin_Cancer
        self.Other_Cancer=Other_Cancer
        self.Depression=Depression
        self.Diabetes=Diabetes
        self.Arthritis=Arthritis
        self.Sex=Sex
        self.Age_Category=Age_Category
        self.Height=Height
        self.Weight=Weight
        self.BMI=BMI
        self.Smoking_History=Smoking_History
        self.Alcohol_Consumption=Alcohol_Consumption
        self.Fruit_Consumption=Fruit_Consumption
        self.Green_Vegetables_Consumption=Green_Vegetables_Consumption
        self.FriedPotato_Consumption=FriedPotato_Consumption
        
    def get_data_as_dataframe(self):
        
        try:
            custom_data_input_dict = {
                'General_Health':[self.General_Health],
                'Checkup':[self.Checkup	],
                'Exercise':[self.Exercise],
                'Skin_Cancer':[self.Skin_Cancer],
                'Other_Cancer':[self.Other_Cancer],
                'Depression':[self.Depression],
                'Diabetes':[self.Diabetes],
                'Arthritis':[self.Arthritis],
                'Sex':[self.Sex],
                'Age_Category':[self.Age_Category],
                'Height':[self.Height],
                'Weight':[self.Weight],
                'BMI':[self.BMI],
                'Smoking_History':[self.Smoking_History],
                'Alcohol_Consumption':[self.Alcohol_Consumption],
                'Fruit_Consumption':[self.Fruit_Consumption],
                'Green_Vegetables_Consumption':[self.Green_Vegetables_Consumption],
                'FriedPotato_Consumption':[self.FriedPotato_Consumption]


            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
            
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)