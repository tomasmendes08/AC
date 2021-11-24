# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:25:49 2021

@author: tomas
"""
from new2 import full_data, full_data_test, all_ids_test, pd, np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score



train_split, test_split = train_test_split(full_data, test_size=0.25, stratify=full_data['status'])

# df_majority = train_split[train_split.status == 1]
# df_minority = train_split[train_split.status == -1]

# df_minority_upsampled = resample(df_minority, 
#                                  replace=True,     # sample with replacement
#                                  n_samples=282    # to match majority class
#                                  )

# train_split = pd.concat([df_majority, df_minority_upsampled])

X_train = train_split.iloc[:, :-1].values
y_train = train_split.iloc[:, -1].values
X_test = test_split.iloc[:, :-1].values
y_test = test_split.iloc[:, -1].values


# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# dt_classifier = DecisionTreeClassifier(random_state=1)


dt_classifier = AdaBoostClassifier()

dt_grid_search = GridSearchCV(dt_classifier,
                            param_grid={},
                            scoring='roc_auc',
                            cv=5)

dt_grid_search.fit(X_train, y_train)
print('Best score: {}'.format(dt_grid_search.best_score_))

print(53 * '=')
print("TRAINING")
predict_dt_train = dt_grid_search.predict(X_train)
print('Precision score: {}'.format(precision_score(y_train, predict_dt_train)))
# print("F1 Score: {}".format(f1_score(y_train, predict_dt_train)))
print(f"ROC: {roc_auc_score(y_train, predict_dt_train)}")
print('\nClassification Report: ')
print(classification_report(y_train, predict_dt_train, labels=np.unique(predict_dt_train)))


print(53 * '=')
print("TESTING")
predict_dt_test = dt_grid_search.predict(X_test)
print('Precision score: {}'.format(precision_score(y_test, predict_dt_test)))
# print('Best parameters: {}'.format(dt_grid_search.best_params_))
# print("F1 Score: {}".format(f1_score(y_test, predict_dt_test)))
print(f"ROC: {roc_auc_score(y_test, predict_dt_test)}")
print('\nClassification Report: ')
print(classification_report(y_test, predict_dt_test, labels=np.unique(predict_dt_test)))


predictions_train = dt_grid_search.predict(X_train)
predictions_test = dt_grid_search.predict(X_test)


predictions_competition = dt_grid_search.predict_proba(full_data_test)
predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted', 'col2'])
predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_test, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')


results.to_csv('out.csv', index = False)
