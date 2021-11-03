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
full_data = data2.merge(loan_train2, how="inner")

clean_data = full_data.drop(columns=['disp_id', 'client_id', 'account_id', 'district_id', 'creation_date', 'loan_date'])
joined_data = clean_data.drop_duplicates(subset=clean_data.columns[1:], keep="last") #props mano caio
final_data = joined_data.drop(columns=['loan_id'])


loan_test2 = loan_test.rename(columns={'date':'loan_date'})
full_data_test = data2.merge(loan_test2, how="inner")

clean_data_test = full_data_test.drop(columns=['disp_id', 'client_id', 'account_id', 'district_id', 'creation_date', 'loan_date'])
joined_data_test = clean_data_test.drop_duplicates(subset=clean_data_test.columns[1:], keep="last")
all_ids_test = joined_data_test['loan_id'].values
final_data_test = joined_data_test.drop(columns=['loan_id', 'status'])



train_split, test_split = train_test_split(final_data, test_size=0.25, stratify=final_data['status'])

X_train = train_split.iloc[:, :-1].values
y_train = train_split.iloc[:, -1].values
X_test = test_split.iloc[:, :-1].values
y_test = test_split.iloc[:, -1].values

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


dt_classifier = DecisionTreeClassifier(random_state=1)

dt_grid_search = GridSearchCV(dt_classifier,
                            param_grid={},
                            scoring='roc_auc',
                            cv=10)

dt_grid_search.fit(X_train, y_train)
predictions_train = dt_grid_search.predict(X_train)
predictions_test = dt_grid_search.predict(X_test)

predictions_competition = dt_grid_search.predict_proba(final_data_test)
predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted', 'col2'])
predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_test, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')


results.to_csv('out.csv', index = False)


