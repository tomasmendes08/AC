# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 17:01:19 2021

@author: pedro
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

#clients = clients.merge(districts, left_on='district_id' , right_on='code')
#clients = clients.merge(accounts, how='inner')
clients = clients.merge(disps, how='inner')

df_majority = loan_train[loan_train.status == 1]
df_minority = loan_train[loan_train.status == -1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=282    # to match majority class
                                 )

loan_train_balanced = pd.concat([df_majority, df_minority_upsampled])

full_data = clients.merge(loan_train_balanced, left_on='account_id', right_on='account_id', how='inner')

full_test_data = clients.merge(loan_test, left_on='account_id', right_on='account_id', how='inner')


account_dic = {'OWNER':1, 'DISPONENT':0}
full_data['type'].replace(account_dic, inplace=True)
full_test_data['type'].replace(account_dic, inplace=True)

clean_data = full_data.drop(columns=['district_id','disp_id'])

clean_test_data = full_test_data.drop(columns=['district_id','disp_id', 'status'])


train_split, test_split = train_test_split(clean_data, random_state=1, test_size=0.25, stratify=clean_data['status'])

X_train = train_split.iloc[:, :-1].values
y_train = train_split.iloc[:, -1].values
X_test = test_split.iloc[:, :-1].values
y_test = test_split.iloc[:, -1].values

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


dt_classifier = DecisionTreeClassifier(random_state=1)

dt_tuned_parameter = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}

dt_grid_search = GridSearchCV(dt_classifier,
                            param_grid=dt_tuned_parameter,
                            scoring='precision_weighted',
                            cv=10)

dt_grid_search.fit(X_train, y_train)
print('Best score: {}'.format(dt_grid_search.best_score_))
print('Best parameters: {}'.format(dt_grid_search.best_params_))



predictions_train = dt_grid_search.predict_proba(X_train)
predictions_test = dt_grid_search.predict_proba(X_test)

all_inputs = clean_data.drop(columns=['status'])
all_labels = clean_data['status'].values

cv_scores = cross_val_score(dt_classifier, all_inputs, all_labels, cv=10, scoring="accuracy")
print(cv_scores)

scaler.fit(clean_test_data)
X_train_comp = scaler.fit_transform(clean_test_data)

all_ids_comp = loan_test['loan_id'].values

predictions_competition = dt_grid_search.predict_proba(clean_test_data)
predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted', 'col2'])
predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_comp, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')

results.to_csv('out.csv', index = False)






'''account_data = pd.read_csv('data/account.csv', na_values=['NA'], sep=';', low_memory=False)
client_data = pd.read_csv('data/client.csv', na_values=['NA'], sep=';', low_memory=False)
disp_data = pd.read_csv('data/disp.csv', na_values=['NA'], sep=';', low_memory=False)
district_data = pd.read_csv('data/district.csv', na_values=['NA'], sep=';', low_memory=False)

card_train_data = pd.read_csv('data/card_train.csv', na_values=['NA'], sep=';', low_memory=False)
# loan_train_data = pd.read_csv('data/loan_train.csv', na_values=['NA'], sep=';', low_memory=False)
# trans_train_data = pd.read_csv('data/trans_train.csv', na_values=['NA'], sep=';', low_memory=False)

# Merging the data

district_data = district_data.rename(columns={"code ":"code"})
print(district_data.columns)

client_data = client_data.merge(district_data, left_on='district_id', right_on='code')
client_data.drop('code', axis=1, inplace=True)'''