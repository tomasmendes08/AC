import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
#from imblearn.under_sampling import RandomUnderSampler

def generate_gender(month):
    
    if month > 12:
        gender = 1
        month = month - 50
    else:
        gender = 0
    
    return gender, month


def convert_date(date):
    year = (date // 10000 + 1900)*365
    days = date % 100
    month_aux = date // 100 % 100
    days += year
    
    gender, month = generate_gender(month_aux)

    return (days + month, gender)  

def process_date(df, col_list, col_name='date'):
    count = 0
    for i in col_list:
        df[i] = df[col_name]
        df[i] = df[i].apply(lambda x : convert_date(x)[count])
        count+=1
    df.drop(columns={col_name}, inplace=True)


def compute_proportions(df, var):
    return df[var].value_counts()


#loading accounts data
accounts = pd.read_csv("./data/account.csv", sep=';')

#process account creation date
process_date(accounts, ['account_creation_date_in_days'])


#loading loans data
loans = pd.read_csv('./data/loan_train.csv', sep=';')
loans['status'].replace({1:0, -1:1}, inplace=True)

#process loan creation date
process_date(loans, ['loan_creation_date_in_days'])

#loading transaction data
transactions = pd.read_csv('./data/trans_train.csv', sep=';')

#process transaction date
process_date(transactions, ['trans_date'])

transactions.drop(columns={'operation', 'k_symbol', 'bank', 'account'}, inplace=True)
transactions['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)

#amount
amount_min = transactions.groupby(['account_id'], as_index=False)['amount'].min()
amount_min.rename(columns={'amount':'min_amount'}, inplace=True)
amount_max = transactions.groupby(['account_id'], as_index=False)['amount'].max()
amount_max.rename(columns={'amount':'max_amount'}, inplace=True)
amount_avg = transactions.groupby(['account_id'], as_index=False)['amount'].mean()
amount_avg.rename(columns={'amount':'avg_amount'}, inplace=True)

#balance
balance_min = transactions.groupby(['account_id'], as_index=False)['balance'].min()
balance_min.rename(columns={'balance':'min_balance'}, inplace=True)
balance_max = transactions.groupby(['account_id'], as_index=False)['balance'].max()
balance_max.rename(columns={'balance':'max_balance'}, inplace=True)
balance_avg = transactions.groupby(['account_id'], as_index=False)['balance'].mean()
balance_avg.rename(columns={'balance':'avg_balance'}, inplace=True)

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


#loading disps data
disps = pd.read_csv("./data/disp.csv", sep=';')

disps_groups = disps.groupby(['account_id'], as_index=False)['type'].count()

trans_disps = trans_processed.merge(disps_groups, on='account_id', how='inner').rename(columns={'type':'members'})


#loading clients data
clients = pd.read_csv("./data/client.csv", sep=';')

clients_disps = clients.merge(disps, on='client_id', how='inner')
owner_disps = clients_disps[clients_disps['type'] == 'OWNER']
owner_disps.drop(columns=['type'], inplace=True)


#processing birth date
process_date(owner_disps, [ 'birthdate' ,'gender'],'birth_number')

#loading districts data
districts = pd.read_csv("./data/district.csv", sep=';')

#replace missing values with the next column's value
districts["unemploymant rate '95 "] = np.where(districts["unemploymant rate '95 "] == '?', districts["unemploymant rate '96 "], districts["unemploymant rate '95 "])
districts["unemploymant rate '95 "] = pd.to_numeric(districts["unemploymant rate '95 "])


#replace missing values with the next column's value
districts["no. of commited crimes '95 "] = np.where(districts["no. of commited crimes '95 "] == '?', districts["no. of commited crimes '96 "], districts["no. of commited crimes '95 "])
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
    recent = trans_group.get_group(key)['trans_date'].max()
    last_balance.append(transactions.query('account_id == ' + str(key) + ' and trans_date == ' + str(recent))['balance'].to_list()[0])
    
last_balance_dataframe = pd.DataFrame({'account_id' : account_ids, 'last_balance' : last_balance})


    
    
loans_merged = loans.merge(trans_disps, on='account_id', how='inner')
loans_merged = loans_merged.merge(last_balance_dataframe, on='account_id', how='inner')
loans_merged = loans_merged.merge(owner_disps, on='account_id', how='inner')
loans_merged = loans_merged.merge(clear_districts, left_on='district_id', right_on='code ', how='inner')


loan_dates = loans_merged['loan_creation_date_in_days']
owners_dates = loans_merged['birthdate']

owner_ages = []

for i in range(len(loan_dates)):
    owner_ages.append(loan_dates[i] - owners_dates[i])
    
loans_merged['owner_age'] = owner_ages
status = loans_merged['status']
loans_merged.drop(columns=['account_id', 'client_id', 'district_id', 'disp_id', 'code ', 'status'], inplace=True)
loans_merged['status'] = status



#loading loans data
loans_test = pd.read_csv('./data/loan_test.csv', sep=';')
loans_test['status'].replace({1:0, -1:1}, inplace=True)


#loading transaction data
transactions_test = pd.read_csv('./data/trans_test.csv', sep=';')

#process loan creation date
process_date(loans_test, ['loan_creation_date_in_days'])

#process transaction date
process_date(transactions_test, ['trans_date'])



transactions_test.drop(columns={'operation', 'k_symbol', 'bank', 'account'}, inplace=True)
transactions_test['type'].replace({'withdrawal in cash':'withdrawal'}, inplace=True)

#amount
amount_min = transactions_test.groupby(['account_id'], as_index=False)['amount'].min()
amount_min.rename(columns={'amount':'min_amount'}, inplace=True)
amount_max = transactions_test.groupby(['account_id'], as_index=False)['amount'].max()
amount_max.rename(columns={'amount':'max_amount'}, inplace=True)
amount_avg = transactions_test.groupby(['account_id'], as_index=False)['amount'].mean()
amount_avg.rename(columns={'amount':'avg_amount'}, inplace=True)

#balance
balance_min = transactions_test.groupby(['account_id'], as_index=False)['balance'].min()
balance_min.rename(columns={'balance':'min_balance'}, inplace=True)
balance_max = transactions_test.groupby(['account_id'], as_index=False)['balance'].max()
balance_max.rename(columns={'balance':'max_balance'}, inplace=True)
balance_avg = transactions_test.groupby(['account_id'], as_index=False)['balance'].mean()
balance_avg.rename(columns={'balance':'avg_balance'}, inplace=True)

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

trans_disps = trans_processed.merge(disps_groups, on='account_id', how='inner').rename(columns={'type':'members'})

trans_group = transactions_test.groupby(['account_id'], as_index=False)
account_ids = []
last_balance = []


for key in trans_group.groups.keys():
    account_ids.append(key)
    recent = trans_group.get_group(key)['trans_date'].max()
    last_balance.append(transactions_test.query('account_id == ' + str(key) + ' and trans_date == ' + str(recent))['balance'].to_list()[0])
    
last_balance_dataframe = pd.DataFrame({'account_id' : account_ids, 'last_balance' : last_balance})

loans_test_merged = loans_test.merge(trans_disps, on='account_id', how='inner')
loans_test_merged = loans_test_merged.merge(last_balance_dataframe, on='account_id', how='inner')
loans_test_merged = loans_test_merged.merge(owner_disps, on='account_id', how='inner')
loans_test_merged = loans_test_merged.merge(clear_districts, left_on='district_id', right_on='code ', how='inner')


loan_dates = loans_test_merged['loan_creation_date_in_days']
owners_dates = loans_test_merged['birthdate']

owner_ages = []

for i in range(len(loan_dates)):
    owner_ages.append(loan_dates[i] - owners_dates[i])
    
loans_test_merged['owner_age'] = owner_ages
#status = loans_test_merged['status']
loans_test_merged.drop(columns=['account_id', 'client_id', 'district_id', 'disp_id', 'code ', 'status'], inplace=True)
#loans_test_merged['status'] = status


train_split, test_split = train_test_split(loans_merged, test_size=0.25, stratify=loans_merged['status'])

df_majority = train_split[train_split.status == 0]
df_minority = train_split[train_split.status == 1]

df_minority_upsampled = resample(df_minority, 
                                  replace=True,     # sample with replacement
                                  n_samples=282    # to match majority class
                                  )

train_split = pd.concat([df_majority, df_minority_upsampled])

#undersample = RandomUnderSampler(sampling_strategy='majority')
#X_over, y_over = undersample.fit_resample(X_train, y_train)

X_train = train_split.iloc[:, :-1].values
y_train = train_split.iloc[:, -1].values
X_test = test_split.iloc[:, :-1].values
y_test = test_split.iloc[:, -1].values


dt_classifier = AdaBoostClassifier(random_state=1)


dt_grid_search = GridSearchCV(dt_classifier,
                            param_grid={},
                            scoring='roc_auc',
                            cv=5)

#tuned_parameters = {'n_estimators': [300],
#                     'max_features': ['auto', 'sqrt'],
#                     'max_depth': [4, 6, 8, 10],
#                     'criterion': ['gini', 'entropy'],
#                     'class_weight': [{0:1, 1:6}]}
#
#dt_grid_search = GridSearchCV(RandomForestClassifier(),
#                     tuned_parameters,
#                     n_jobs=-1,
#                     scoring='roc_auc',
#                     cv=3)


all_ids_test = loans_test_merged['loan_id'].values

dt_grid_search.fit(X_train, y_train)
best_score = dt_grid_search.best_score_
print("Best Score: " + str(best_score))
print('Best parameters: {}'.format(dt_grid_search.best_params_))
predictions_train = dt_grid_search.predict(X_train)
predictions_test = dt_grid_search.predict(X_test)


predictions_competition = dt_grid_search.predict_proba(loans_test_merged)

print("Area under ROC curve: " + str(roc_auc_score(y_test, dt_grid_search.predict(X_test))))

predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted','col2'])
#predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_test, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')


results.to_csv('out.csv', index = False)