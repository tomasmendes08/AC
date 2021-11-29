# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 20:15:02 2021

@author: pedro
"""

import pandas as pd
import numpy as np

def generate_gender(month):
    gender = month > 12
    if gender: month = month - 50
    
    return gender, month

def convert_date(date):
    year = date // 10000 + 1900
    days = date % 100
    month_aux = date // 100 % 100
    
    gender, month = generate_gender(month_aux)

    return (year, month, days, gender)  

def process_date(df, col_list, col_name='date'):
    count = 0
    for i in col_list:
        df[i] = df[col_name]
        df[i] = df[i].apply(lambda x : convert_date(x)[count])
        count+=1
    df.drop(columns={col_name}, inplace=True)

#loading clients data
clients = pd.read_csv("./data/client.csv", sep=';')

#processing birth date
process_date(clients, ['byear', 'bmonth', 'bday', 'gender'],'birth_number')

#loading disps data
disps = pd.read_csv("./data/disp.csv", sep=';')

#processing disps data
disps['type'].replace({'OWNER':True, 'DISPONENT':False}, inplace=True)

#rename type column to disp_type
disps.rename(columns={'type':'disp_type'}, inplace=True)

#merging clients with disps
clients_disps = clients.merge(disps, on="client_id", how="inner")

#loading cards data
cards = pd.read_csv("./data/card_train.csv", sep=';')
# cards.drop(columns=["type"],inplace=True)

#rename type column to card_type
cards.rename(columns={'type':'card_type'}, inplace=True)
cards.drop(columns=["issued"], inplace = True)

#process cards data
# process_date(cards, ['cyear', 'cmonth', 'cday'], 'issued')

#merging clients and disps with cards
clients_disps_cards = clients_disps.merge(cards, on='disp_id', how='outer')

clients_disps_cards["card_id"] = clients_disps_cards["card_id"].fillna(0)
clients_disps_cards["card_type"] = clients_disps_cards["card_type"].fillna("Undefined")


#loading districts data
districts = pd.read_csv("./data/district.csv", sep=';')

#clear districts data
districts.rename(columns={'name ':'name', 'code ':'district_id'}, inplace=True)
districts.drop(columns=["name", "region"], inplace=True)


#replace missing values with the next column's value
districts["unemploymant rate '95 "] = np.where(districts["unemploymant rate '95 "] == '?', districts["unemploymant rate '96 "], districts["unemploymant rate '95 "])
districts["unemploymant rate '95 "] = pd.to_numeric(districts["unemploymant rate '95 "])


#replace missing values with the next column's value
districts["no. of commited crimes '95 "] = np.where(districts["no. of commited crimes '95 "] == '?', districts["no. of commited crimes '96 "], districts["no. of commited crimes '95 "])
districts["no. of commited crimes '95 "] = pd.to_numeric(districts["no. of commited crimes '95 "])

#merging clients, disps and cards with districts
clients_disps_cards_districts = clients_disps_cards.merge(districts, on="district_id", how='inner')


#loading accounts data
accounts = pd.read_csv("./data/account.csv", sep=';')

#processing account creation date and frequency

accounts.rename(columns={'district_id':'acc_district_id'}, inplace=True)
process_date(accounts,['ayear', 'amonth', 'aday'])

accounts = pd.concat([accounts.drop('frequency', axis=1), pd.get_dummies(accounts['frequency'])], axis=1)

#merging clients, disps, cards and districts with accounts
clients_disps_cards_districts_accounts = clients_disps_cards_districts.merge(accounts, on="account_id")

#loading transaction data
transactions = pd.read_csv('./data/trans_train.csv', sep=';')

#processing transaction data
transactions.drop(columns={'k_symbol','bank','account', 'operation'}, inplace=True)
transactions['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)
transactions['type'] = transactions['type'].apply(lambda x : x == 'credit')
transactions.rename(columns={'type':'trans_type'}, inplace=True)

process_date(transactions, ['tyear', 'tmonth', 'tday'])

#merging clients, disps, cards, districts and accounts with transactions
clients_disps_cards_districts_accounts_transactions = clients_disps_cards_districts_accounts.merge(transactions, on="account_id", how="inner")


#loading loan data
loans = pd.read_csv('./data/loan_train.csv', sep=';')

#processing loan data
loans.rename(columns={'amount':'loan_amount', 'duration':'loan_duration', 'payments':'loan_monthly_payment'}, inplace=True)
process_date(loans, ['lyear', 'lmonth', 'lday'])
loans = loans[['loan_id', 'account_id', 'loan_amount', 'loan_duration', 'loan_monthly_payment', 'lyear', 'lmonth', 'lday', 'status']]

#merging clients, disps, cards, districts, accounts and transactions with loans
full_data = clients_disps_cards_districts_accounts_transactions.merge(loans, on="account_id")

#groupby loan_id
full_data = full_data.groupby('loan_id').mean()     #provavelmente, isto não é uma boa solução

# Create correlation matrix
corr_matrix = full_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
full_data.drop(to_drop, axis=1, inplace=True)

#check which columns have null values
# nan_values = full_data.isna()
# nan_columns = nan_values.any()
# columns_with_nan = full_data.columns[nan_columns].tolist()


# ---------------------------------------------------------------------------


#preprocessing test data

#loading cards data
cards_test = pd.read_csv("./data/card_test.csv", sep=';')
# cards.drop(columns=["type"],inplace=True)

#rename type column to card_type
cards_test.rename(columns={'type':'card_type'}, inplace=True)
cards_test.drop(columns=["issued"], inplace = True)

#process cards data
# process_date(cards_test, ['cyear', 'cmonth', 'cday'], 'issued')

#merging clients and disps with cards_test
clients_disps_cards_test = clients_disps.merge(cards_test, on='disp_id', how='outer')

clients_disps_cards_test["card_id"] = clients_disps_cards_test["card_id"].fillna(0)
clients_disps_cards_test["card_type"] = clients_disps_cards_test["card_type"].fillna("Undefined")

#merging clients, disps and cards_test with districts
clients_disps_cards_districts_test = clients_disps_cards_test.merge(districts, on="district_id", how='inner')

#merging clients, disps, cards_test and districts with accounts
clients_disps_cards_districts_accounts_test = clients_disps_cards_districts_test.merge(accounts, on="account_id")

#loading transaction test data
transactions_test = pd.read_csv('./data/trans_test.csv', sep=';')

#processing transaction test data
transactions_test.drop(columns={'k_symbol','bank','account', 'operation'}, inplace=True)
transactions_test['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)
transactions_test['type'] = transactions_test['type'].apply(lambda x : x == 'credit')
transactions_test.rename(columns={'type':'trans_type'}, inplace=True)

process_date(transactions_test, ['tyear', 'tmonth', 'tday'])

#merging clients, disps, cards_test, districts and accounts with transactions_test
clients_disps_cards_districts_accounts_transactions_test = clients_disps_cards_districts_accounts_test.merge(transactions_test, on="account_id")


#loading loan test data
loans_test = pd.read_csv('./data/loan_test.csv', sep=';')
all_ids_test = loans_test['loan_id'].values

#processing loan test data
loans_test.rename(columns={'amount':'loan_amount', 'duration':'loan_duration', 'payments':'loan_monthly_payment'}, inplace=True)
process_date(loans_test, ['lyear', 'lmonth', 'lday'])
loans_test = loans_test[['loan_id', 'account_id', 'loan_amount', 'loan_duration', 'loan_monthly_payment', 'lyear', 'lmonth', 'lday', 'status']]

#merging clients, disps, cards_test, districts, accounts and transactions_test with loans_test
full_data_test = clients_disps_cards_districts_accounts_transactions_test.merge(loans_test, on="account_id")
full_data_test.drop(columns={"status"}, inplace=True)

#groupby loan_id
full_data_test = full_data_test.groupby('loan_id').mean()

full_data_test.drop(to_drop, axis=1, inplace=True)

# # Create correlation matrix
# corr_matrix = full_data_test.corr().abs()

# # Select upper triangle of correlation matrix
# upper_test = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# # Find features with correlation greater than 0.95
# to_drop_test = [column for column in upper_test.columns if any(upper_test[column] > 0.95)]

# # Drop features 
# full_data_test.drop(to_drop_test, axis=1, inplace=True)

