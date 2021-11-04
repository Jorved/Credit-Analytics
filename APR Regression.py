#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 22:53:12 2021

@author: joshuathomas
"""

from sklearn import linear_model
import pandas as pd

df = pd.read_csv('/Users/joshuathomas/Desktop/cs_training.csv')



import datetime as dt

df['birth_date'] = pd.to_datetime(df['birth_date'])
df['birth_date']=df['birth_date'].map(dt.datetime.toordinal)

X = df[['birth_date', 'credit_score','month_since_default_adjusted','open_credit_cards','employment_status',]]
X = pd.get_dummies(data=X, drop_first=True)
y = df['apr_offer_selected']

X.head()

regr = linear_model.LinearRegression()
regr.fit(X, y)


print(regr.coef_)

print(regr.intercept_)

# redicted = regr.predict([[35177, 723,2,72,1 ]])

# print(predicted)
