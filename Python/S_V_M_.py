# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:58:39 2019

@author: RAMAN
"""

import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\RAMAN\\Documents\\Python Scripts\\Data")
os.getcwd()

train_data = pd.read_csv('R_Module_Day_7.2_Credit_Risk_Train_data.csv')
test_data = pd.read_csv('R_Module_Day_8.2_Credit_Risk_Test_data.csv')

train_data['Source'] = 'Train'
test_data['Source'] = 'Test'
raw_data = pd.concat([train_data, test_data], axis = 0)
raw_data.shape

raw_data.dtypes
raw_data.isnull().sum()

raw_data['Dependents'].unique()
raw_data['Dependents'] = np.where(raw_data['Dependents'] == '3+', '3', raw_data['Dependents'])

# raw_data['Dependents'] = pd.factorize( [raw_data['Dependents']] )[0]

# raw_data['Dependents'].astype("category").cat.codes
# raw_data['Dependents'] = pd.Categorical()

raw_data['Dependents'] = pd.Series(raw_data['Dependents'], dtype = 'float64')

for i in raw_data.columns:
    if(raw_data[i].dtype == object):
        raw_data[i] = raw_data[i].fillna('Unknown')
        
    else:
        temp_imputation_value = raw_data.loc[raw_data['Source'] == "Train", i].median()
        raw_data[i].fillna(temp_imputation_value, inplace = True)

categ_vars = raw_data.loc[:, raw_data.dtypes == object].columns

dummy_df = pd.get_dummies(raw_data[categ_vars].drop(['Loan_ID', 'Source', 'Loan_Status'], axis=1), drop_first = True, dtype = int)
dummy_df.columns

full_data = pd.concat([raw_data, dummy_df], axis = 1)
Cols_To_Drop = categ_vars.drop(['Source','Loan_Status'])

full_data.shape
full_data2 = full_data.drop(Cols_To_Drop, axis = 1).copy()
full_data2.info()

description = full_data2.describe()

full_data2['Loan_Status']=np.where(full_data2['Loan_Status']== 'N',1,0)

fullraw3 = full_data2.copy()



var_list_for_detection = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

def outliers(df, var_list):
    for i in var_list:
        quant = i.quantile([0.25, 0.75])
        upper_bound = quant[0.75] + 1.5 * (quant[0.75] - quant[0.25])
        lower_bound = quant[0.25] - 1.5 * (quant[0.75] - quant[0.25])
        
        df[i] = np.where(df[i] > upper_bound, upper_bound, df[i])
        df[i] = np.where(df[i] < lower_bound, lower_bound, df[i])
    return (df)


def outlier_detection_correction(variable_list, data):
    for varName in variable_list:
        Q3 = np.percentile(data.loc[data['Source'] == 'Train', varName], 75)
        Q1 = np.percentile(data.loc[data['Source'] == 'Train', varName], 25)
        
        upper = Q3 + 1.5*(Q3-Q1)
        lower = Q1 - 1.5*(Q3-Q1)

        data[varName] = np.where(data[varName] > upper, upper, data[varName])
        data[varName] = np.where(data[varName] < lower, lower, data[varName])
    
    return (data)

before_summary = fullraw3[fullraw3['Source'] == 'Train'].describe()
Fullraw = outlier_detection_correction(var_list_for_detection, fullraw3)
#Fullraw = fullraw3.copy()
after_summary = Fullraw[fullraw3['Source'] == 'Train'].describe()

#####################################################
# Sampling - Train_X, Train_Y, Test_X, Test_Y
#####################################################

Train_X = Fullraw.loc[Fullraw['Source'] == 'Train'].drop(['Source', 'Loan_Status'], axis = 1).copy()

Train_Y = Fullraw.loc[Fullraw['Source'] == 'Train']
Train_Y = Train_Y['Loan_Status'].copy()
Train_X.shape

Test_X = Fullraw.loc[Fullraw['Source'] == 'Test'].drop(['Source', 'Loan_Status'], axis = 1).copy()

Test_Y = Fullraw.loc[Fullraw['Source'] == 'Test']
Test_Y = Test_Y['Loan_Status'].copy()
Test_X.shape

##########################################
# Model 2
##########################################

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

M2 = SVC() # Default kernel is 'rbf'

M2_Model = M2.fit(Train_X, Train_Y)
Test_class_M2 = M2_Model.predict(Test_X)
Confusion_Mat2 = confusion_matrix(Test_Y, Test_class_M2)
sum(np.diagonal(Confusion_Mat2))/Test_X.shape[0] * 100 # 79%

f1_score(Test_Y, Test_class_M2)

#########################
# Model 1 (kernel = linear)
# DO NOT TRY THIS AT HOME
#########################
 

#############################
# Manual Grid Searching
#############################

#############################
# grid Search - Alternate 1
#############################
df = {}
dataTable = pd.DataFrame()
cost = [1,2]
gamma = [0.01, 0.02]
kernel = ['sigmoid', 'rbf']

for krnl in kernel:
    for ct in cost:
        for gmma in gamma:
            M1 = SVC(kernel=krnl, C = ct, gamma = gmma)

            M1_Model = M1.fit(Train_X, Train_Y)
            Test_class_M1 = M1_Model.predict(Test_X)
            Confusion_Mat1 = confusion_matrix(Test_Y, Test_class_M1)
            accuracy = sum(np.diagonal(Confusion_Mat1))/Test_X.shape[0] * 100 
            df = pd.DataFrame({"Cost" : [ct],
                               "Kernel" : [krnl],
                               "Gamma" : [gmma],
                               "accuracy" : [accuracy]})
            dataTable = dataTable.append(df, ignore_index = True)
            

########################################
# python way of doing this( grid search )
########################################

from sklearn.model_selection import GridSearchCV

my_param_grid = {'C' : [1,2], 'gamma': [0.01, 0.1], 'kernel': ['sigmoid', 'rbf']} # it's a dictionary
SVM_GS = GridSearchCV(SVC(), param_grid= my_param_grid, scoring = 'accuracy', cv = 3)

SVM_GS_Model = SVM_GS.fit(Train_X, Train_Y)

# #Result in dictionary format
# SVM_GS_Model.cv_results_
# # Best tuning parameters
# SVM_GS_Model.best_params_

# Results in dataframe
SVM_Grid_Search_Df = pd.DataFrame.from_dict(SVM_GS_Model.cv_results_)
