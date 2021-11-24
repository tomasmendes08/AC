# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:25:49 2021

@author: tomas
"""
from new import full_data, full_data_test, all_ids_test, pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


train_split, test_split = train_test_split(full_data, test_size=0.25, stratify=full_data['status'])

X_train = train_split.iloc[:, :-1].values
y_train = train_split.iloc[:, -1].values
X_test = test_split.iloc[:, :-1].values
y_test = test_split.iloc[:, -1].values


# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# dt_classifier = DecisionTreeClassifier(random_state=1)
dt_classifier = AdaBoostClassifier(random_state=1)

dt_grid_search = GridSearchCV(dt_classifier,
                            param_grid={},
                            scoring='roc_auc',
                            cv=5)

dt_grid_search.fit(X_train, y_train)
best_score = dt_grid_search.best_score_
print("Best Score: " + str(best_score))

predictions_train = dt_grid_search.predict(X_train)
predictions_test = dt_grid_search.predict(X_test)


predictions_competition = dt_grid_search.predict_proba(full_data_test)
predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted', 'col2'])
predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_test, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')


results.to_csv('out.csv', index = False)
