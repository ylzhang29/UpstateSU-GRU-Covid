import os

os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.preprocessing import MinMaxScaler
import pickle

import csv
from math import sqrt

from tensorflow import keras

import keras
import tensorflow as tf


from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras import optimizers
from keras.optimizers import SGD, Adam, Adamax, Nadam, Adagrad
from keras.optimizers import Adadelta, RMSprop
# from hyperas.distributions import choice, uniform
from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import TerminateOnNaN

tn = TerminateOnNaN()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, GRU, LSTM, Reshape, Lambda
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import random
import os

with open('County_Demo_01102021.csv', newline='') as csvfile:
    demo = pd.read_csv(csvfile)
    demo_id = demo.iloc[:, :3]
    print(demo.columns, demo.shape)
    print(demo_id, demo_id.shape)
    values = demo.drop(['combined_key', 'fips', 's_county'], axis=1).values
    scaled_demo = MinMaxScaler().fit_transform(values)
    scaled_demo = pd.DataFrame(scaled_demo)
    print(scaled_demo.shape)
    demo = demo_id.join(scaled_demo, on=None)
	demo = demo.drop(['combined_key'], axis = 1)
    print(demo.columns, "demographic data", demo.shape)

# read the ID keys
id = demo.filter(['s_county'], axis=1).drop_duplicates()
print("In Demo file: unique numbers of region names", len(id['s_county'].unique()))

### READ long DATASET Covid ###
with open('JH_0411.csv', newline='') as csvfile:
    covid = pd.read_csv(csvfile)

covid = covid.drop(['combined_key', 'total_days_cases', 'total_days_mb'], axis=1)
covid.isna().sum(axis=0)
print(covid.shape)
covid = covid.dropna(subset=['google_mb'], axis=0)

print(covid.columns, covid.shape, "Total numbers of regions in Covid data:", len(covid['s_county'].unique()))

# obtain covid data that also have demographic information
covid = id.merge(covid, on="s_county", how='inner')
covid = covid.sort_values(['s_county', 'days'], axis=0)
print("Unique numbers of region  in Covid Dataset that has demographic data:", len(covid['s_county'].unique()))
covid.isna().sum(axis=0)

print("Num in Testset", len(covid[covid['testset'] == 0]['s_county'].unique()))
print("Num in Train", len(covid[covid['testset'] == -1]['s_county'].unique()))
print("Num in Valid", len(covid[covid['testset'] == 2]['s_county'].unique()))

###########################################################################################
new_incidence = covid.dropna(subset=['new_cases7'], axis =0)
new_incidence['new_cases7ma']=new_incidence['new_cases7'].astype('Int64', errors = "ignore")
new_incidence['new_deaths7ma']=new_incidence['new_deaths7'].astype('Int64', errors = "ignore")
new_incidence.to_csv("covid_new_incidence.csv", index=False, header=True)

print(new_incidence.shape, new_incidence.columns)
print(covid.shape, covid.columns)

name = "United States"
county = new_incidence[new_incidence['s_county'] == name]
# print(county[['testset', 'days', 'Date', 'cases', 'new_cases', 'deaths', 'new_deaths']].iloc[30:40])

print(new_incidence.columns, new_incidence.shape, "Total numbers of regions in Covid data:",
      len(new_incidence['s_county'].unique()))
new_incidence.isna().sum(axis=0)

# plot county data
from datetime import datetime

dates = county['Date'].values
x_values = [datetime.strptime(d, "%d%b%Y").date() for d in dates]

plt.figure(figsize=(9, 8))
plt.suptitle("{}".format(name))
plt.plot(x_values, county['new_cases'].values, 'go')
plt.plot(x_values, county['new_cases7'].values, 'r-')
plt.plot(x_values, county['new_cases7ma'].values, 'k.')
plt.title("Daily New Cases")
plt.show()

plt.figure(figsize=(9, 8))
plt.suptitle("{}".format(name))
plt.plot(x_values, county['new_deaths'].values, 'go')
plt.plot(x_values, county['new_deaths7'].values, 'r-')
plt.title("Daily New Deaths")
plt.show()

plt.figure(figsize=(15, 10))
plt.suptitle("{}".format(name))
plt.plot(x_values, county['google_mb'].values, 'go')
plt.plot(x_values, county['gmb7'].values, 'r-')
plt.title("Daily Average Google Mobility")
plt.show()


#################################################### Calculate R0 ####################################################
new_incidence = pd.read_csv("covid_new_incidence.csv")#, converters = {'new_cases7':int, 'new_deaths7': int})
# new_incidence = new_incidence[new_incidence['fips']==0]

def R_fun(incidence):
    r = ro.r
    # r.source(path+"rtest.R")
    r.source("R0_Estimate.R")
    p = r.rtest(incidence)
    R = ro.DataFrame(p)
    return R


import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri

pandas2ri.activate()

covid2 = pd.DataFrame()  # columns = ['s_county', 'fips', 'testset', 'Date', 'date', 'days', 'cases', 'deaths', 'google_mb', 'r0'])
i = 0
for county in new_incidence['s_county'].unique():
    i = i + 1
    print(i, "County Name:", county)
    # if i <= 2474:  # skipped the i=972 and 2313 county that was giving error in R and interrupting the loop
    #     continue
    county = new_incidence[new_incidence['s_county'] == county]
    subset = county.drop(['s_county', 'fips', 'testset', 'Date', 'date', 'new_cases', 'new_deaths', 'google_mb' ,'new_cases7', 'new_deaths7'], axis=1)
    subset['Date'] = county['days'].astype('Int64')
    subset['I'] = county['new_cases7ma'].astype('Int64', errors = 'ignore')
    subset['I'][subset['I']<0]=0
    subset = subset[['Date', 'I']]
    cases = pandas2ri.py2rpy(subset)
    r0 = R_fun(cases)
    r0_est = pd.DataFrame(np.asarray(r0[0]), dtype='int', columns=['days'])
    r0_est['r0'] = pd.DataFrame(np.asarray(r0[2]))
    subset = county.merge(r0_est, on='days', how='inner')
    subset['total_days'] = pd.DataFrame(np.repeat((subset.shape[0]), (subset.shape[0]), axis=0), dtype='int')
    print(subset)
    covid2 = subset.append(covid2, ignore_index=True)

covid2.to_csv("covid2_R0.csv", index=False, header=True)

covid2 = covid2.sort_values(['s_county', 'days'], axis=0)
covid2 = pd.read_csv("covid2_R0.csv")

print(covid2[['s_county', 'days', 'r0']])
print(covid2.columns, covid2.shape, "Total numbers of regions that has R values:", len(covid2['s_county'].unique()))
print("Num in Testset", len(covid2[covid2['testset'] == 0]['s_county'].unique()))
print("Num in Train", len(covid2[covid2['testset'] == -1]['s_county'].unique()))
print("Num in Valid", len(covid2[covid2['testset'] == 2]['s_county'].unique()))


###plot some county's R0
county = covid2[covid2['s_county'] == "United States"]
# print(county[['days', 'cases']])

# plot scaled data vs original data to check if scalling was done right
from datetime import datetime

dates = county['Date'].values
x_values = [datetime.strptime(d, "%d%b%Y").date() for d in dates]

plt.figure(figsize=(9, 4))
plt.plot(x_values, county['r0'].values, 'yo')
plt.title('R Values: {}'.format(county))
plt.show()



