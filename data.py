import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, classification_report, SCORERS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from dateutil.relativedelta import relativedelta
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

import datetime

def generate_gender(month):
    
    if month > 12:
        gender = 1
        month = month - 50
    else:
        gender = 0
    
    return gender, month


def convert_date(date):
    year = date // 10000 + 1900
    days = date % 100
    month = date // 100 % 100
    
    gender, month = generate_gender(month)
    
    date = datetime.datetime(year, month, days)

    return (date.date(), gender)  

def process_date(df, col_list, col_name='date'):
    count = 0
    for i in col_list:
        df[i] = df[col_name]
        df[i] = df[i].apply(lambda x : convert_date(x)[count])
        count+=1
    df.drop(columns={col_name}, inplace=True)


def statistics(df, column, extra=""):
    minimum = df.groupby(['account_id'], as_index=False)[column].min()
    minimum.rename(columns={column:'min_' + extra + column}, inplace=True)
    maximum = df.groupby(['account_id'], as_index=False)[column].max()
    maximum.rename(columns={column:'max_'+ extra + column}, inplace=True)
    average = df.groupby(['account_id'], as_index=False)[column].mean()
    average.rename(columns={column:'avg_'+ extra + column}, inplace=True)
    
    return minimum, maximum, average



#loading accounts data
accounts = pd.read_csv("./data/account.csv", sep=';')

#process account creation date
process_date(accounts, ['account_creation_date'])


#loading loans data
loans = pd.read_csv('./data/loan_train.csv', sep=';')
loans['status'].replace({1:0, -1:1}, inplace=True)

#process loan creation date
process_date(loans, ['loan_creation_date'])

#loading transaction data
transactions = pd.read_csv('./data/trans_train.csv', sep=';', low_memory=False)

#process transaction date
process_date(transactions, ['transaction_date'])

transactions.drop(columns={'operation', 'k_symbol', 'bank', 'account'}, inplace=True)
transactions['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)

#amount
amount_min, amount_max, amount_avg = statistics(transactions, 'amount')

#balance
balance_min, balance_max, balance_avg = statistics(transactions, 'balance')


#credit
credit_min, credit_max, credit_avg = statistics(transactions.loc[transactions['type'] == 'credit'], 'amount', "credit_")

#withdrawal
withdrawal_min, withdrawal_max, withdrawal_avg = statistics(transactions.loc[transactions['type'] == 'withdrawal'], 'amount', "withdrawal_")

movements = transactions.groupby(['account_id'], as_index=False)['type'].count()
types = transactions.groupby(['account_id'], as_index=False)['type']
credit = []
withdrawal = []

for key in types.groups.keys():
    credit.append(types.get_group(key).value_counts().to_dict()['credit'])
    if('withdrawal' not in types.get_group(key).value_counts().to_dict().keys()):
        withdrawal.append(0)
    else:
      withdrawal.append(types.get_group(key).value_counts().to_dict()['withdrawal'])  
      
trans_processed = amount_min.merge(amount_max, on='account_id', how='inner')
trans_processed = trans_processed.merge(amount_avg, on='account_id', how='inner')
trans_processed = trans_processed.merge(balance_min, on='account_id', how='inner')
trans_processed = trans_processed.merge(balance_max, on='account_id', how='inner')
trans_processed = trans_processed.merge(balance_avg, on='account_id', how='inner')
trans_processed['credit'] = credit
trans_processed['withdrawal'] = withdrawal
trans_processed = trans_processed.merge(credit_min, on='account_id', how='left')
trans_processed = trans_processed.merge(credit_max, on='account_id', how='left')
trans_processed = trans_processed.merge(credit_avg, on='account_id', how='left')
trans_processed = trans_processed.merge(withdrawal_min, on='account_id', how='left')
trans_processed = trans_processed.merge(withdrawal_max, on='account_id', how='left')
trans_processed = trans_processed.merge(withdrawal_avg, on='account_id', how='left')
trans_processed.fillna(0, inplace=True)


#loading disps data
disps = pd.read_csv("./data/disp.csv", sep=';')

disps_groups = disps.groupby(['account_id'], as_index=False)['type'].count()

trans_disps = trans_processed.merge(disps_groups, on='account_id', how='inner').rename(columns={'type':'members'})


#loading clients data
clients = pd.read_csv("./data/client.csv", sep=';')

clients_disps = clients.merge(disps, on='client_id', how='inner')
owner_disps = clients_disps.loc[clients_disps.type == 'OWNER']
owner_disps.drop(columns=['type'], inplace=True)


#processing birth date
process_date(owner_disps, ['birthdate' ,'gender'],'birth_number')

#loading districts data
districts = pd.read_csv("./data/district.csv", sep=';')

#replace missing values with the next column's value
districts["unemploymant rate '95 "] = np.where(districts["unemploymant rate '95 "] == '?',
                                                districts["unemploymant rate '96 "],
                                                districts["unemploymant rate '95 "])
districts["unemploymant rate '95 "] = pd.to_numeric(districts["unemploymant rate '95 "])


#replace missing values with the next column's value
districts["no. of commited crimes '95 "] = np.where(districts["no. of commited crimes '95 "] == '?',
                                                    districts["no. of commited crimes '96 "],
                                                    districts["no. of commited crimes '95 "])
districts["no. of commited crimes '95 "] = pd.to_numeric(districts["no. of commited crimes '95 "])

districts.drop(columns={'name '}, inplace=True)
  
binary_frequency = pd.get_dummies(districts["region"])
clear_districts = pd.concat((districts, binary_frequency), axis=1)
clear_districts.drop(columns="region", inplace=True)


trans_group = transactions.groupby(['account_id'], as_index=False)
account_ids = []
last_balance = []


for key in trans_group.groups.keys():
    account_ids.append(key)
    recent = trans_group.get_group(key)['transaction_date'].max()
    aux = transactions[transactions['transaction_date'] == recent]
    last_balance.append(min(aux[aux['account_id'] == key]['balance'].tolist()))
    
last_balance_dataframe = pd.DataFrame({'account_id' : account_ids,
                                        'last_balance' : last_balance})


#loading cards data
cards = pd.read_csv("./data/card_train.csv", sep=';')

cards_disps = cards.merge(disps, on='disp_id', how='inner')

number_cards = cards_disps.groupby(['account_id'], as_index=False).size()
number_cards.rename(columns={'size':'number_of_cards'}, inplace=True)
       
    
loans_merged = loans.merge(trans_disps, on='account_id', how='left')
loans_merged = loans_merged.merge(last_balance_dataframe, on='account_id', how='inner')
loans_merged = loans_merged.merge(owner_disps, on='account_id', how='inner')
loans_merged = loans_merged.merge(clear_districts, left_on='district_id', right_on='code ', how='inner')
loans_merged = loans_merged.merge(number_cards, on='account_id', how='left')
loans_merged['number_of_cards'].replace(np.nan,0, inplace=True)


loan_dates = loans_merged['loan_creation_date']
owners_dates = loans_merged['birthdate']

owner_ages = []

for i in range(len(loan_dates)):
    owner_ages.append(relativedelta(loan_dates[i], owners_dates[i]).years)

    
loans_merged['owner_age'] = owner_ages    
    
loans_creation_dates = loans_merged['loan_creation_date']
years = []
months = []
days = []

for date in loans_creation_dates:
    years.append(date.year) 
    months.append(date.month) 
    days.append(date.day) 
   
   
loans_merged['loan_year'] = years
loans_merged['loan_month'] = months
loans_merged['loan_day'] = days 

account_ids = loans_merged['account_id'].to_list()

status = loans_merged['status']
loans_merged.drop(columns=['account_id', 'client_id', 'district_id', 'disp_id', 'code ', 'status', 'loan_creation_date', 'birthdate', 'loan_id'], inplace=True)
loans_merged['status'] = status



#loading loans data
loans_test = pd.read_csv('./data/loan_test.csv', sep=';')
loans_test['status'].replace({1:0, -1:1}, inplace=True)


#loading transaction data
transactions_test = pd.read_csv('./data/trans_test.csv', sep=';')

#process loan creation date
process_date(loans_test, ['loan_creation_date'])

#process transaction date
process_date(transactions_test, ['transaction_date'])



transactions_test.drop(columns={'operation', 'k_symbol', 'bank', 'account'}, inplace=True)
transactions_test['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)

#amount
amount_min, amount_max, amount_avg = statistics(transactions_test, 'amount')


#balance
balance_min, balance_max, balance_avg = statistics(transactions_test, 'balance')


#credit
credit_min, credit_max, credit_avg = statistics(transactions_test.loc[transactions_test['type'] == 'credit'], 'amount', "credit_")

#withdrawal
withdrawal_min, withdrawal_max, withdrawal_avg = statistics(transactions_test.loc[transactions_test['type'] == 'withdrawal'], 'amount', "withdrawal_")

movements = transactions_test.groupby(['account_id'], as_index=False)['type'].count()
types = transactions_test.groupby(['account_id'], as_index=False)['type']
credit = []
withdrawal = []

for key in types.groups.keys():
    credit.append(types.get_group(key).value_counts().to_dict()['credit'])
    if('withdrawal' not in types.get_group(key).value_counts().to_dict().keys()):
        withdrawal.append(0)
    else:
      withdrawal.append(types.get_group(key).value_counts().to_dict()['withdrawal'])  
      
trans_processed = amount_min.merge(amount_max, on='account_id', how='inner')
trans_processed = trans_processed.merge(amount_avg, on='account_id', how='inner')
trans_processed = trans_processed.merge(balance_min, on='account_id', how='inner')
trans_processed = trans_processed.merge(balance_max, on='account_id', how='inner')
trans_processed = trans_processed.merge(balance_avg, on='account_id', how='inner')
trans_processed['credit'] = credit
trans_processed['withdrawal'] = withdrawal
trans_processed = trans_processed.merge(credit_min, on='account_id', how='left')
trans_processed = trans_processed.merge(credit_max, on='account_id', how='left')
trans_processed = trans_processed.merge(credit_avg, on='account_id', how='left')
trans_processed = trans_processed.merge(withdrawal_min, on='account_id', how='left')
trans_processed = trans_processed.merge(withdrawal_max, on='account_id', how='left')
trans_processed = trans_processed.merge(withdrawal_avg, on='account_id', how='left')
trans_processed.fillna(0, inplace=True)

trans_disps = trans_processed.merge(disps_groups, on='account_id', how='inner').rename(columns={'type':'members'})

trans_group = transactions_test.groupby(['account_id'], as_index=False)
account_ids = []
last_balance = []


for key in trans_group.groups.keys():
    account_ids.append(key)
    recent = trans_group.get_group(key)['transaction_date'].max()
    aux = transactions_test[transactions_test['transaction_date'] == recent]
    last_balance.append(min(aux[aux['account_id'] == key]['balance'].tolist()))
    
last_balance_dataframe = pd.DataFrame({'account_id' : account_ids, 'last_balance' : last_balance})


#loading cards data
cards_test = pd.read_csv("./data/card_test.csv", sep=';')

cards_disps_test = cards_test.merge(disps, on='disp_id', how='inner')

number_cards_test = cards_disps_test.groupby(['account_id'], as_index=False).size()
number_cards_test.rename(columns={'size':'number_of_cards'}, inplace=True)


loans_test_merged = loans_test.merge(trans_disps, on='account_id', how='inner')
loans_test_merged = loans_test_merged.merge(last_balance_dataframe, on='account_id', how='inner')
loans_test_merged = loans_test_merged.merge(owner_disps, on='account_id', how='inner')
loans_test_merged = loans_test_merged.merge(clear_districts, left_on='district_id', right_on='code ', how='inner')
loans_test_merged = loans_test_merged.merge(number_cards_test, on='account_id', how='left')
loans_test_merged['number_of_cards'].replace(np.nan,0, inplace=True)


loan_dates = loans_test_merged['loan_creation_date']
owners_dates = loans_test_merged['birthdate']

owner_ages = []

for i in range(len(loan_dates)):
    owner_ages.append(relativedelta(loan_dates[i], owners_dates[i]).years)
    
loans_test_merged['owner_age'] = owner_ages


loans_creation_dates = loans_test_merged['loan_creation_date']
years = []
months = []
days = []

for date in loans_creation_dates:
    years.append(date.year) 
    months.append(date.month) 
    days.append(date.day) 
   
   
loans_test_merged['loan_year'] = years
loans_test_merged['loan_month'] = months
loans_test_merged['loan_day'] = days 

account_ids = loans_test_merged['account_id'].to_list()

all_ids_test = loans_test_merged['loan_id'].values

loans_test_merged.drop(columns=['account_id', 'client_id', 'district_id', 'disp_id', 'code ', 'status', 'loan_creation_date', 'birthdate'], inplace=True)

loans_merged.to_csv("./data/train.csv", index = False)
loans_test_merged.to_csv("./data/test.csv", index = False)