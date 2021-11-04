#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:23:36 2021

@author: joshuathomas
"""

# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing



applications = pd.read_csv('/Users/joshuathomas/Desktop/applications.csv',parse_dates=(True))

print(applications.shape)
print(applications.head())

applications[['Applications_made']].plot(title='Application Data')

decompose_result = seasonal_decompose(applications['Applications_made'],model='a',period=4)
decompose_result.plot()


applications['HWES3_ADD'] = ExponentialSmoothing(applications['Applications_made'],trend='add',seasonal='add',seasonal_periods=4).fit().fittedvalues
applications['HWES3_MUL'] = ExponentialSmoothing(applications['Applications_made'],trend='mul',seasonal='mul',seasonal_periods=4).fit().fittedvalues
applications[['Applications_made','HWES3_ADD','HWES3_MUL']].plot(title='Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality');