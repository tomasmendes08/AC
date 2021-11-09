# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 20:15:02 2021

@author: pedro
"""

import pandas as pd

#loading clients data
clients = pd.read_csv("./data/client.csv", sep=';')

#loading disps data
disps = pd.read_csv("./data/disp.csv", sep=';')

#merging clients with disps
clients_disps = clients.merge(disps, on="client_id", how="inner")

#loading cards data
cards = pd.read_csv("./data/card_train.csv", sep=';')

#rename type column to disp_type
disps.rename(columns={'type ':'disp_type'}, inplace=True)

#rename type column to card_type
cards.rename(columns={'type':'card_type'}, inplace=True)

#merging clients and disps with cards
clients_disps_cards = clients_disps.merge(cards, on='disp_id', how='outer')

#loading districts data
districts = pd.read_csv("./data/district.csv", sep=';')

#clear districts data
districts.rename(columns={'name ':'name', 'code ':'district_id'}, inplace=True)
districts.drop(columns=["name", "region"], inplace=True)

districts_without_nulls = districts[districts["unemploymant rate '95 "] != '?']
unemployment_rates_95 = pd.to_numeric(districts_without_nulls["unemploymant rate '95 "])

#mean of the unemploymant rate '95
mean_unemployment_rate_95 = unemployment_rates_95.mean()

#replace missing values for the mean. Maybe use the minimal or the maximal value?
districts["unemploymant rate '95 "].replace('?', str(round(mean_unemployment_rate_95, 2)), inplace=True)
districts["unemploymant rate '95 "] = pd.to_numeric(districts["unemploymant rate '95 "])


districts_without_nulls = districts[districts["no. of commited crimes '95 "] != '?']
commited_rates_95 = pd.to_numeric(districts_without_nulls["no. of commited crimes '95 "])

#mean of the commited crimes in '95
mean_commited_rates_95 = commited_rates_95.mean()

#replace missing values for the mean. Maybe use the minimal or the maximal value?
districts["no. of commited crimes '95 "].replace('?', str(round(mean_commited_rates_95, 2)), inplace=True)
districts["no. of commited crimes '95 "] = pd.to_numeric(districts["no. of commited crimes '95 "])

