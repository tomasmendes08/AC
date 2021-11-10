# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 20:15:02 2021

@author: pedro
"""

import pandas as pd
import numpy as np

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


#replace missing values with the next column's value
districts["unemploymant rate '95 "] = np.where(districts["unemploymant rate '95 "] == '?', districts["unemploymant rate '96 "], districts["unemploymant rate '95 "])
districts["unemploymant rate '95 "] = pd.to_numeric(districts["unemploymant rate '95 "])


#replace missing values with the next column's value
districts["no. of commited crimes '95 "] = np.where(districts["no. of commited crimes '95 "] == '?', districts["no. of commited crimes '96 "], districts["no. of commited crimes '95 "])
districts["no. of commited crimes '95 "] = pd.to_numeric(districts["no. of commited crimes '95 "])

#merging clients, disps and cards with districts
clients_disps_cards_districts = clients_disps_cards.merge(districts, on="district_id")


#loading accounts data
accounts = pd.read_csv("./data/account.csv", sep=';')

def generate_gender(month):
    gender = month > 12
    if gender: month -= 50
    
    return gender

def convert_date(date):
    year = date // 10000
    days = date % 100
    month = date // 100 % 100
    
    gender = generate_gender(month)
    
    return (year, month, days, gender)
    
