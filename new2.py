# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 23:59:55 2021

@author: tomas
"""
import pandas as pd
import numpy as np
import datetime

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
    
def process_date2(df):
    
    for index, row in df.iterrows():
        year, month, day, gender = convert_date(df.loc[index, "date"])
        days = (year * 365) + (month * 30) + day
        df.loc[index, "date"] = days
        # print(index)

def calculate_account_age(df):
    
    for index, row in df.iterrows():
        x = datetime.datetime(int(row["lyear"]), int(row["lmonth"]), int(row["lday"]))
        y = datetime.datetime(int(row["ayear"]), int(row["amonth"]), int(row["aday"]))
        diff = str(x-y)
        
        df.loc[index, "account_age"] = int(diff.split(' ')[0]) 
    
    return df
        

def calculate_amount_balance_values(df):
    
    for index, row in df.iterrows():
        transaction_rows = df[df["account_id"]==df.loc[index,"account_id"]]
        
        
        most_recent = max(transaction_rows['date'])
        oldest = min(transaction_rows['date'])
        diff = most_recent - oldest
        
        df.loc[index, "min_amount"] = transaction_rows["amount"].min()
        df.loc[index, "max_amount"] = transaction_rows["amount"].max()
        df.loc[index, "mean_amount"] = round(transaction_rows["amount"].mean(), 2)
        df.loc[index, "range_amount"] = transaction_rows["amount"].max() - transaction_rows["amount"].min()
        df.loc[index, "total_amount"] = round(transaction_rows["amount"].sum(), 2)
        df.loc[index, "amount_monthly"] = transaction_rows["amount"].sum() / (diff/30.0)
        
        df.loc[index, "min_balance"] = transaction_rows["balance"].min()
        df.loc[index, "max_balance"] = transaction_rows["balance"].max()
        df.loc[index, "mean_balance"] = round(transaction_rows["balance"].mean(), 2)
        df.loc[index, "range_balance"] = transaction_rows["balance"].max() - transaction_rows["balance"].min()
        
        df.loc[index, "can_pay"] = df.loc[index, 'amount_monthly'] > df.loc[index, 'loan_monthly_payment']
        # print(index)
    
    return df


def calculate_withdrawals_credits(df):
    
    for index, row in df.iterrows():
        transaction_rows = df[df["account_id"]==df.loc[index,"account_id"]]
        
        df.loc[index, "nmr_withdrawals"] = len(transaction_rows[transaction_rows["trans_type"]=="withdrawal"])
        df.loc[index, "nmr_credits"] = len(transaction_rows[transaction_rows["trans_type"]=="credit"])
        
    return df

def process_client_age(df):
    for index, row in df.iterrows():
        x = datetime.datetime(int(row["lyear"]), int(row["lmonth"]), int(row["lday"]))
        y = datetime.datetime(int(row["byear"]), int(row["bmonth"]), int(row["bday"]))
        
        diff = str(x-y)
        
        df.loc[index, "client_age"] = round(int(diff.split(' ')[0]) / 365.25,2)
        
    return df


#read loans
loans = pd.read_csv("./data/loan_train.csv", sep=";")
loans['status'].replace({-1:1, 1:0}, inplace=True)

#process loans
loans.rename(columns={'amount':'loan_amount', 'duration':'loan_duration', 'payments':'loan_monthly_payment'}, inplace=True)
loan_date = ['lyear', 'lmonth', 'lday']
process_date(loans, loan_date)

#loading accounts data
accounts = pd.read_csv("./data/account.csv", sep=';')

#processing account creation date and frequency

accounts.rename(columns={'district_id':'acc_district_id'}, inplace=True)
account_date = ['ayear', 'amonth', 'aday']
process_date(accounts, account_date)

accounts = pd.concat([accounts.drop('frequency', axis=1), pd.get_dummies(accounts['frequency'])], axis=1)

#merging loans with accounts
loans_accounts = loans.merge(accounts, on="account_id")
calculate_account_age(loans_accounts)

#loading transaction data
transactions = pd.read_csv('./data/trans_train.csv', sep=';')

#processing transaction data
transactions.drop(columns={'k_symbol','bank','account', 'operation'}, inplace=True)
transactions['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)
transactions['type'] = transactions['type'].apply(lambda x : x == 'credit')
transactions.rename(columns={'type':'trans_type'}, inplace=True)
loans_accounts_transactions = loans_accounts.merge(transactions, on="account_id")
process_date2(loans_accounts_transactions)

loans_accounts_transactions["nmr_of_movements"] = loans_accounts_transactions.groupby('account_id')['account_id'].transform('count')
calculate_amount_balance_values(loans_accounts_transactions)
calculate_withdrawals_credits(loans_accounts_transactions)


#loading clients data
clients = pd.read_csv("./data/client.csv", sep=';')

#processing birth date
process_date(clients, ['byear', 'bmonth', 'bday', 'gender'],'birth_number')

#loading disps data
disps = pd.read_csv("./data/disp.csv", sep=';')

#processing disps data
disps['type'].replace({'OWNER':1, 'DISPONENT':0}, inplace=True)

#rename type column to disp_type
disps.rename(columns={'type':'disp_type'}, inplace=True)
# disps.drop(columns={'type'}, inplace=True)

clients_disps = clients.merge(disps, on="client_id")
clients_disps["nmr_of_members"] = clients_disps.groupby('account_id')['account_id'].transform('count')


#merging clients and disps with loans_accounts_transactions
loans_accounts_transactions_clients_disps = loans_accounts_transactions.merge(clients_disps, on="account_id")
process_client_age(loans_accounts_transactions_clients_disps)

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



full_data = loans_accounts_transactions_clients_disps.merge(districts, on="district_id")

full_data = full_data.drop_duplicates(subset=['loan_id'], keep='first')

# Create correlation matrix
# corr_matrix = full_data.corr().abs()

# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# # Drop features 
# full_data.drop(to_drop, axis=1, inplace=True)



##############################     TEST     #########################################


# read loans_test
loans_test = pd.read_csv("./data/loan_test.csv", sep=";")
loans_test['status'].replace({-1:1, 1:0}, inplace=True)
all_ids_test = loans_test['loan_id'].values

#process loans_test
loans_test.rename(columns={'amount':'loan_amount', 'duration':'loan_duration', 'payments':'loan_monthly_payment'}, inplace=True)
process_date(loans_test, ['lyear', 'lmonth', 'lday'])

#merging loans with accounts
loans_accounts_test = loans_test.merge(accounts, on="account_id")
calculate_account_age(loans_accounts_test)

#loading transaction data
transactions_test = pd.read_csv('./data/trans_test.csv', sep=';')

#processing transaction data
transactions_test.drop(columns={'k_symbol','bank','account', 'operation'}, inplace=True)
transactions_test['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)
transactions_test['type'] = transactions_test['type'].apply(lambda x : x == 'credit')
transactions_test.rename(columns={'type':'trans_type'}, inplace=True)
loans_accounts_transactions_test = loans_accounts_test.merge(transactions_test, on="account_id")
process_date2(loans_accounts_transactions_test)

loans_accounts_transactions_test["nmr_of_movements"] = loans_accounts_transactions_test.groupby('account_id')['account_id'].transform('count')
calculate_amount_balance_values(loans_accounts_transactions_test)
calculate_withdrawals_credits(loans_accounts_transactions_test)


#merging clients and disps with loans_accounts_transactions
loans_accounts_transactions_clients_disps_test = loans_accounts_transactions_test.merge(clients_disps, on="account_id")
process_client_age(loans_accounts_transactions_clients_disps_test)


full_data_test = loans_accounts_transactions_clients_disps_test.merge(districts, on="district_id")

full_data_test = full_data_test.drop_duplicates(subset=['loan_id'], keep='first')

