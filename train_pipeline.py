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

account_dic = {'OWNER':1, 'DISPONENT':0}
frequency_dic = {'monthly issuance':2, 'weekly issuance':1, 'issuance after transaction':0}


disps['type'].replace(account_dic, inplace=True)
accounts['frequency'].replace(frequency_dic, inplace=True)

data1 = disps.merge(accounts, how="inner")

data2 = data1.rename(columns={'date':'creation_date'})
loan_train2 = loan_train.rename(columns={'date':'loan_date'})
full_data = data2.merge(loan_train2, how="inner")

clean_data = full_data.drop(columns=['disp_id', 'client_id', 'account_id', 'district_id', 'type'])

