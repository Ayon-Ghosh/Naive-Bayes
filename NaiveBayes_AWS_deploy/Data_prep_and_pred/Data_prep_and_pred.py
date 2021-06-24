# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import os
import json
from flask import Flask, render_template, request,jsonify
import pandas as pd
import numpy as np
import math
import csv
import pickle


class batch_data_pred:
    def __init__(self, dataframe):
        self.dataframe = dataframe



    def nan_feature(self, dataframe):
        #cols = dataframe.columns
        #print(cols)
        #fill_Nan_columns = [i for i in cols if i !='Pregnancies' or 'Outcome']
        columns = list(dataframe.columns)
        fill_Nan_columns = columns[1:-2]
        dataframe[fill_Nan_columns] = dataframe[fill_Nan_columns].replace(0, np.nan)
        df_to_impute = dataframe[fill_Nan_columns]
        return df_to_impute

    def impute_data(self, dataframe, df_to_impute):
        df_to_impute['BMI'] = df_to_impute['BMI'].fillna(dataframe['BMI'].mean())
        BMI_test = pd.DataFrame(df_to_impute.loc[pd.isnull(df_to_impute.SkinThickness)]['BMI'])

        LM = pickle.load(open('regression_imputer.sav', 'rb'))
        SkinThickness_test = LM.predict(BMI_test)

        SkinThickness_test_df = pd.DataFrame(data=SkinThickness_test, columns=['SkinThickness'],
                                             index=BMI_test.index.copy())
        df_to_impute = pd.merge(df_to_impute, SkinThickness_test_df, how='left', left_index=True, right_index=True)
        df_to_impute["SkinThickness_y"] = df_to_impute["SkinThickness_y"].fillna(df_to_impute["SkinThickness_x"])
        df_to_impute = df_to_impute.drop('SkinThickness_x', axis=1)
        df_to_impute = df_to_impute.rename(columns={'SkinThickness_y': 'SkinThickness'})
        df_to_impute['Glucose'] = df_to_impute['Glucose'].replace(0, df_to_impute['Glucose'].mean())

        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=100, verbose=0)
        imp.fit(df_to_impute)
        imputed_data = imp.transform(df_to_impute)
        df_to_impute = pd.DataFrame(imputed_data, columns=df_to_impute.columns)
        df_to_impute = df_to_impute[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']]
        # dataframe[[fill_Nan_columns]] =  df_to_impute
        return df_to_impute

    def scaler_transform(self, dataframe):
        scaler = pickle.load(open('standardScalar.sav', 'rb'))
        tranformed_array = scaler.transform(dataframe)
        tranformed_dataframe = pd.DataFrame(tranformed_array, columns=dataframe.columns)
        return tranformed_dataframe

    def batch_predict(self, tranformed_dataframe):
        loaded_model = pickle.load(open('modelForPrediction.sav', 'rb'))
        prediction = loaded_model.predict(tranformed_dataframe)
        return prediction


class single_pred:

    def single_predict(self, dict_pred):
        with open("standardScalar.sav", 'rb') as f:
            scalar = pickle.load(f)

        with open("modelForPrediction.sav", 'rb') as f:
            model = pickle.load(f)
        data_df = pd.DataFrame(dict_pred,index=[1,])
        scaled_data = scalar.transform(data_df)
        predict = model.predict(scaled_data)
        #predict = model.predict(data_df)
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'

        return result
