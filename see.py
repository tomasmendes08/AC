# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:46:37 2021

@author: pedro
"""

import pandas as pd


out = pd.read_csv("./out.csv", sep=',')

ids = out['loan_id'].tolist()
aucs = out['confidence(false)']

final_csv = pd.DataFrame(ids, columns=['Id'])
final_csv['Predicted'] = aucs

final_csv.to_csv('final.csv', index = False)