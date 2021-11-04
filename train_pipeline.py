# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:58:34 2021

@author: tomas
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from collections import Counter

from sklearn.utils import resample


clients = pd.read_csv("./data/client.csv", sep=';')
districts = pd.read_csv("./data/district.csv", sep=';')
accounts = pd.read_csv("./data/account.csv", sep=';')
disps = pd.read_csv("./data/disp.csv", sep=';')
loan_train = pd.read_csv("./data/loan_train.csv", sep=';')
loan_test = pd.read_csv("./data/loan_test.csv", sep=';')
districts = districts.rename(columns={'code ':'code'})

account_dic = {'OWNER':1, 'DISPONENT':2}
frequency_dic = {'monthly issuance':2, 'weekly issuance':1, 'issuance after transaction':0}


disps['type'].replace(account_dic, inplace=True)
accounts['frequency'].replace(frequency_dic, inplace=True)

data1 = disps.merge(accounts, how="inner")

data2 = data1.rename(columns={'date':'creation_date'})
loan_train2 = loan_train.rename(columns={'date':'loan_date'})
data3 = data2.merge(clients, how='inner')
full_data = data3.merge(loan_train2, how="inner")

clean_data = full_data.drop(columns=['disp_id', 'client_id', 'account_id', 'district_id'])
joined_data = clean_data.drop_duplicates(subset=clean_data.columns[1:], keep="last") #props mano caio

creation_dates = joined_data['creation_date'].values
loan_dates = joined_data['loan_date'].values
birth_dates = joined_data['birth_number'].values

final_data = joined_data.drop(columns=['loan_id', 'creation_date', 'loan_date', 'birth_number'])



genders = []
ages = []

for date in creation_dates:    
    ages.append(2021 - (1900 + date // 10000))
    
final_data['account_age'] = ages


ages.clear()

for date in loan_dates:
    ages.append(2021 - (1900 + date // 10000))
    
final_data['loan_age'] = ages

ages.clear()

for date in birth_dates:    
    if date % 10000 // 100 > 12:
        genders.append(0)
    else:
        genders.append(1)
        
    ages.append(2021 - (1900 + date // 10000))
    
    
final_data['client_age'] = ages  
final_data['gender'] = genders
    
    
loan_test2 = loan_test.rename(columns={'date':'loan_date'})
full_data_test = data3.merge(loan_test2, how="inner")

clean_data_test = full_data_test.drop(columns=['disp_id', 'client_id', 'account_id', 'district_id'])
joined_data_test = clean_data_test.drop_duplicates(subset=clean_data_test.columns[1:], keep="last")
all_ids_test = joined_data_test['loan_id'].values

creation_dates = joined_data_test['creation_date'].values
loan_dates = joined_data_test['loan_date'].values
birth_dates = joined_data_test['birth_number'].values

final_data_test = joined_data_test.drop(columns=['loan_id', 'status', 'creation_date', 'loan_date', 'birth_number'])


ages = []

for date in creation_dates:    
    ages.append(2021 - (1900 + date // 10000))
    
final_data_test['account_age'] = ages


ages.clear()

for date in loan_dates:
    ages.append(2021 - (1900 + date // 10000))
    
final_data_test['loan_age'] = ages

ages.clear()
genders.clear()

for date in birth_dates:    
    if date % 10000 // 100 > 12:
        genders.append(0)
    else:
        genders.append(1)
        
    ages.append(2021 - (1900 + date // 10000))
    
    
final_data_test['client_age'] = ages  
final_data_test['gender'] = genders

final_data = final_data[['type', 'frequency','amount', 'duration', 'payments', 'account_age','loan_age', 'client_age', 'gender', 'status']]

train_split, test_split = train_test_split(final_data, test_size=0.25, stratify=final_data['status'])


X_train = train_split.iloc[:, :-1].values
y_train = train_split.iloc[:, -1].values
X_test = test_split.iloc[:, :-1].values
y_test = test_split.iloc[:, -1].values


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

svm_classifier = SVC(random_state=1, probability=True)

'''tuned_parameters = [{'kernel': ['rbf', 'linear','poly','sigmoid'], 
                        'gamma': ['auto','scale', 1e-3, 1e-4], 
                        'C': [0.01, 0.1, 1, 10, 100],
                        'class_weight': ['balanced', None]}]'''

grid_search = GridSearchCV(svm_classifier,
                          param_grid={},
                          scoring='roc_auc',
                          n_jobs=-1,
                          cv=10)

# dt_classifier = DecisionTreeClassifier(random_state=1, class_weight={1:1, -1:6})


# dt_grid_search = GridSearchCV(dt_classifier,
#                             param_grid={},
#                             scoring='roc_auc',
#                             cv=10)

grid_search.fit(X_train, y_train)
predictions_train = grid_search.predict(X_train)
predictions_test = grid_search.predict(X_test)

predictions_competition = grid_search.predict_proba(final_data_test)
predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted', 'col2'])
predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_test, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')


results.to_csv('out.csv', index = False)


