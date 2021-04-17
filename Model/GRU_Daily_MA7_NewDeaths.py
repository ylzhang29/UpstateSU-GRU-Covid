import os

os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.preprocessing import MinMaxScaler
import pickle

import csv
from math import sqrt

from tensorflow import keras
import keras
import tensorflow as tf

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

###read and scale demo file
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
print(covid[['Date', 'date']])

# obtain covid data that also have demographic information
covid = id.merge(covid, on="s_county", how='inner')
covid = covid.sort_values(['s_county', 'days'], axis=0)
print("Unique numbers of region  in Covid Dataset that has demographic data:", len(covid['s_county'].unique()))
covid.isna().sum(axis=0)

print("Num in Testset", len(covid[covid['testset'] == 0]['s_county'].unique()))
print("Num in Train set", len(covid[covid['testset'] == -1]['s_county'].unique()))
print("Num in Valid set", len(covid[covid['testset'] == 2]['s_county'].unique()))

##############################################
#load data with R0
covid2= pd.read_csv("covid2_R0_ffilled.csv")
print(covid2.columns)
covid2 = covid2.drop(['Unnamed: 0'], axis = 1)

##############################################
# generate the date up to 1130 (22249) for prediction
# generate the date up to 1207 (22256) for prediction
# generate the date up to 1214 (22263) for prediction

# covid2 = covid2[covid2['date']<=22263]
##############################################

##############################################
# load scaled data
##############################################
scaled_covid= pd.read_csv("scaled_covid.csv")
# scaled_covid = scaled_covid.drop(['Unnamed: 0'], axis = 1)

# #check if scalling was done right
county = scaled_covid[scaled_covid['s_county']=="New York_New York City"]
print(county, county.columns)
#plot scaled data vs original data to check if scalling was done right
plt.figure(figsize=(9, 6))
plt.subplot(221)
plt.plot(county['r0'].values, county['scaled_r0'].values, 'go' )#color='red', marker='o', markersize=12)
#plt.yscale('log')
plt.title('Onondaga scaled vs original r0')
plt.subplot(222)
plt.plot(county['gmb7'].values, county['scaled_mb'].values, 'ro' )
plt.title('Onondaga scaled vs original Google_mb')
plt.subplot(223)
plt.plot(county['new_cases7ma'].values, county['scaled_cases'].values, 'go' )#color='red', marker='o', markersize=12)
plt.title('Onondaga scaled vs original cases')
plt.subplot(224)
plt.plot(county['new_deaths7ma'].values, county['scaled_deaths'].values, 'ro' )
plt.title('Onondaga scaled vs original Deaths')
plt.show()
#

####################################################
###################### Feature Numbers #################
#SET SEQUENCE steps AND FORCAST time steps
#select the data to be sequenced-formatted
steps_past = 90
steps_forw = 30
cols = 3 # total numbers of integer columns ahead of the sequence: fips, testset, date, days,
seq_features = 4  # 4: total numbers of features to reformat to sequence :  'scaled_cases', 'scaled_deaths', 'scaled_mb', 'scaled_r0'
demo_features = 28 # pc and other demo that has no missing values
######################################################
######################################################
#### Generate wide-sequence data from the long data.
def gen_seqdata(sequences, demographic, n_steps, forcast):
    X = np.empty([0, (cols+n_steps+forcast)+demo_features, (seq_features)], dtype = int)
    for i in range(len(sequences)):
        end_ix = i + n_steps
        end_iy = i + n_steps + forcast
        #print(i, end_ix, end_iy)
        if end_iy > len(sequences):
            continue
        # gather input and output parts of the pattern
        data = sequences[:, -(seq_features):]
        id = sequences[:, :-(seq_features)]
        #print('original id shape', id.shape)
        seq = data[i:end_iy].reshape(1, (n_steps+forcast), (seq_features))
        #print("seq data shape", seq.shape)
        id = id[i:i+1].reshape(1, cols)
        # id contains: fips, testset, date, days,
        id = np.hstack((id[:, :2], demographic[:,1:], id[:, 2:]))
        #print("demo", demographic, "demo shape", demographic.shape)
        print( "id shape", id.shape)
        id = np.repeat(id[ :,:, None],seq_features, axis = 2)
        #print("new ID shape", id.shape)
        seq_data =np.concatenate((id, seq), axis = 1)
        #print("new seq shape", seq_data.shape)
        print(X.shape, "seq data shape", seq_data.shape)
        X=np.vstack((X, seq_data))
    return X
#########################################################
## load seq data
seq_X_train = np.load("seq_X_train_7maNewCases.npy", allow_pickle=True)
seq_X_valid = np.load("seq_X_valid_7maNewCases.npy", allow_pickle=True)
seq_X_test = np.load("seq_X_test_7maNewCases.npy", allow_pickle=True)

data_train, data_valid, data_test=seq_X_train, seq_X_valid, seq_X_test
print(data_train.shape, data_valid.shape, seq_X_test.shape)

# SELECT FEATURES AND TARGET FROM DATA
x_train = data_train[:,2:-(steps_forw), :]
#y_train = data_train[:, -(steps_forw):].reshape(-1,1)
# y_train = data_train[:, -(steps_forw):, :]
y_train = data_train[:, -(steps_forw):, 1].reshape(-1, steps_forw,1)
print("training features and targets", x_train.shape, y_train.shape)

x_valid = data_valid[:, 2:-(steps_forw),:]
#y_valid = data_valid[:, -(steps_forw):, :]
y_valid = data_valid[:, -(steps_forw):, 1].reshape(-1, steps_forw,1)
print('validation features and targets', x_valid.shape, y_valid.shape)

x_test = data_test[:, 2:-(steps_forw),:]
#y_test = data_test[:, -(steps_forw):, :]
y_test = data_test[:, -(steps_forw):, 1].reshape(-1, steps_forw,1)
print('Test features and targets', x_test.shape, y_test.shape)


days_data_input_train = x_train.reshape(x_train.shape[0],x_train.shape[1],(seq_features)).astype("float32")
days_data_predict_train = y_train.reshape(y_train.shape[0],y_train.shape[1],1).astype("float32")
days_data_input_validation = x_valid.reshape(x_valid.shape[0],x_valid.shape[1],(seq_features)).astype("float32")
days_data_predict_validation = y_valid.reshape(y_valid.shape[0],y_valid.shape[1],1).astype("float32")
days_data_input_test = x_test.reshape(x_test.shape[0],x_test.shape[1],(seq_features)).astype("float32")
days_data_predict_test = y_test.reshape(y_test.shape[0],y_test.shape[1],1).astype("float32")

predict_zeros_train=np.zeros(days_data_predict_train.shape).astype("float32")
predict_zeros_validation=np.zeros(days_data_predict_validation.shape).astype("float32")
predict_zeros_test=np.zeros(days_data_predict_test.shape).astype("float32")

########################################################################################################

keras.backend.clear_session()

layers = [90, 90] # Number of hidden neuros in each layer of the encoder and decoder
learning_rate = 0.0025261164756672167
decay = .0008232243718101066 # Learning rate decay
optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)
loss = "mse" # Other loss functions are possible, see Keras documentation.
lambda_regulariser = 4.363216857418235e-07 # Will not be used if regulariser is None
regulariser = keras.regularizers.l2(lambda_regulariser)  #None # Possible regulariser: keras.regularizers.l2(lambda_regulariser)
batch_size = 180

num_input_features = seq_features # The dimensionality of the input at each time step. In this case a 1D signal.
num_output_features = 1 # The dimensionality of the output at each time step. In this case a 1D signal.
input_sequence_length = demo_features+steps_past+2 # Length of the sequence used by the encoder
target_sequence_length = steps_forw # Length of the sequence predicted by the decoder
num_steps_to_predict = input_sequence_length  # Length to use when testing the model

encoder_inputs = keras.layers.Input(shape=(None, num_input_features))
encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(keras.layers.GRUCell(hidden_neurons, kernel_regularizer=regulariser, recurrent_regularizer=regulariser, bias_regularizer=regulariser, recurrent_dropout=0))
encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)
print(encoder_outputs_and_states[0].shape,encoder_outputs_and_states[1].shape)
encoder_states = encoder_outputs_and_states[1:]

decoder_inputs = keras.layers.Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(keras.layers.GRUCell(hidden_neurons, kernel_regularizer=regulariser, recurrent_regularizer=regulariser, bias_regularizer=regulariser, recurrent_dropout=0))
decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)
decoder_outputs = decoder_outputs_and_states[0]

decoder_dense = keras.layers.Dense(num_output_features, activation='linear', kernel_regularizer=regulariser, bias_regularizer=regulariser, )
decoder_outputs = decoder_dense(decoder_outputs)


model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.compile(optimizer=optimiser, loss=loss)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)

history = model.fit([days_data_input_train,predict_zeros_train], days_data_predict_train,
                    batch_size=batch_size, epochs=200,
                    validation_data=([days_data_input_validation,predict_zeros_validation],days_data_predict_validation),
                    callbacks=[es,tn])
model.save('Model_deaths_0411.h5')

valid_eval= model.evaluate([days_data_input_validation,predict_zeros_validation],days_data_predict_validation, verbose=2)
train_eval= model.evaluate([days_data_input_train,predict_zeros_train], days_data_predict_train, verbose=0)
print("train mse:", train_eval, "valid mse", valid_eval)

y_pred = model.predict([days_data_input_validation,predict_zeros_validation])
print(y_pred, y_pred.shape)


from scipy import stats
for i in range(y_pred.shape[2]): # google_mg, r0, cases, deaths
    print("Target %d"%(i+1))
    for j in range(y_pred.shape[1]):#12days of predictions
        corr, p = stats.pearsonr(days_data_predict_validation[:, j, i], y_pred[:, j, i])
        mse = np.mean((days_data_predict_validation[:,j,i]-y_pred[:,j,i])**2)
        print("day %d"%(j+1), "mse", mse, "correlation and p", corr, p)

train_loss=np.array(history.history['loss'])
valid_loss=np.array(history.history['val_loss'])
epoch = np.linspace(1,len(train_loss), num = len(train_loss))

# plot loss during training
from matplotlib import pyplot
pyplot.title('MSE')
pyplot.plot(valid_loss, label='valid')
pyplot.plot(train_loss, label='train')
pyplot.legend()
pyplot.show()


###############################
# Predict future 30 days for all counties
####################run predictions on testset
test_all = scaled_covid
print("total", scaled_covid.shape)
print("Unique numbers of region  in Test set:", len(test_all['s_county'].unique()))
print(test_all.columns, test_all.shape)
test_all.isna().sum(axis = 0)

df_test = test_all
df_test = df_test.drop(['s_county',  'date', 'total_days',"Date"], axis = 1)
print(df_test.columns, df_test.shape)

seq_X_test = np.empty([0, (cols+steps_past+demo_features), (seq_features)], dtype = int)
print(seq_X_test.shape)
Pred_test = np.empty([0,steps_forw,seq_features], dtype = float)
Prediction_all = pd.DataFrame()

for county in df_test['fips'].unique():
    print("fips:", county)
    county_data = df_test[df_test['fips'] == county]
    cases, deaths = county_data['new_cases7ma'].values.reshape(-1,1), county_data['new_deaths7ma'].values.reshape(-1,1)
    mb, r0 = county_data['gmb7'].values.reshape(-1,1),county_data['r0'].values.reshape(-1,1)
    data = county_data.drop(['new_cases7ma', 'new_deaths7ma', 'gmb7', 'r0'], axis = 1)
    print("county data", data.columns,  data.shape)
    data=data.values

    c_demo = demo[demo['fips']==county]
    c_name = c_demo['s_county'].unique()
    print(c_name)
    c_demo=c_demo.drop(['s_county'], axis = 1).values
    if c_demo.shape[0]==0:
        print('dropped county:', county)
        continue
    seq_x=gen_seqdata(data, c_demo, (steps_past), 0)
    print("county seq data", seq_x.shape)
    if seq_x.shape[0]==0:
        print("Dropped county", county)
        continue

    ########use this block if generate seriels of prediction and use the mean###########
    days_data_input_test = seq_x[:, 2:, :].astype("float32")
    #days_data_predict_test = seq_x[-1, -(steps_forw):, :].reshape(1, -1, seq_features)
    print("Prediction input", days_data_input_test.shape)
    predict_zeros_test = np.zeros((days_data_input_test.shape[0], (steps_forw), 1)).astype("float32")
    #print(days_data_input_test.shape)
    #print(predict_zeros_test.shape)
    prediction = model.predict([days_data_input_test, predict_zeros_test])
    print(prediction.shape)
    start_day = (days_data_input_test[0, -(steps_past + 1), 0] + steps_past)
    print(start_day)
    DP = np.empty((0, 2))
    # pred_step = (prediction.shape[0]-1)
    pred_step = 1
    for d in np.arange(0, prediction.shape[0], pred_step):
        p = prediction[d, :, :].reshape(30, 1)
        dp = np.hstack((np.arange((start_day), (start_day + steps_forw)).reshape(-1, 1), p))
        DP = np.append(DP, dp, axis=0)
        print(p.shape, dp.shape, DP.shape)
        start_day = start_day + pred_step

    DP = pd.DataFrame(DP, columns=['days', 'pred_deaths'])
    DP.sort_values('days')
    DP_mean = DP.groupby('days').mean()
    print(DP_mean.index)
    p_cases = MinMaxScaler().fit(deaths).inverse_transform(DP_mean['pred_deaths'].values.reshape(-1, 1))
    DP_final = pd.DataFrame(p_cases, columns=['pred_deaths'])  # , 'pred_deaths', 'pred_mb', 'pred_r0'] )
    DP_final['days'] = DP_mean.index
    fips = county_data['fips'].unique()
    fips = np.repeat(fips[0], len(DP_final)).reshape(-1, 1)
    DP_final['fips'] = fips
    DP_final = county_data.merge(DP_final, on=['fips', 'days'], how='outer')
    Prediction_all = Prediction_all.append(DP_final)


print(Prediction_all, Prediction_all.shape, Prediction_all.columns)
print(Prediction_all[['days', 'new_deaths7ma','pred_deaths']][-30:])

# read daily new cases and deaths
covid_new = pd.read_csv("covid2_R0.csv")
print(covid_new.columns)
Date_data = covid_new[['s_county', 'fips', 'Date', 'days', 'new_cases', 'new_deaths','new_cases7', 'new_deaths7']]
Prediction_all = Date_data.merge(Prediction_all, on=['fips', 'days'], how='outer')
Prediction_all.to_csv("Prediction_all_newdeaths_0411.csv", header=True, index=False)

Prediction_all = pd.read_csv("Prediction_all_newdeaths_0411.csv")
print(Prediction_all.columns)
name = "New York_Onondaga"
F = 36067
County = Prediction_all[Prediction_all['fips'] == F]
County = County[County['days'] > 150]
from datetime import datetime

start_date = datetime.strptime(County['Date'].values[0], "%d%b%Y").date()
dates = pd.date_range(start_date, periods=len(County), freq='d')

plt.figure(figsize=(10, 8))
plt.suptitle("County:{}".format(name))
#plt.subplot(2, 2, 1)
plt.title('Predicted vs Actual Deaths')
plt.plot(dates,County['new_deaths'], 'go')
plt.plot(dates, County['new_deaths7ma'], 'm-')
plt.plot(dates, County['pred_deaths'], 'ro')
plt.xticks(rotation=20)
plt.show()

##############################################################################
##### run predictions on all with dropout for N iterations
from datetime import datetime
from keras.models import Model, Sequential
from keras import backend as K
from tensorflow.keras.models import load_model
import keras
import pandas as pd
import numpy as np

def create_dropout_predict_function(model, dropout):
    """
    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers
    Returns
    predict_with_dropout : keras function for predicting with dropout
    """
    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for i in range(len(conf['layers'][2]['config']['cell']['config']['cells'])):
        print(i, "encoder layer")
        # layer['config']["recurrent_dropout"] = dropout
        conf['layers'][2]['config']['cell']['config']['cells'][i]['config']['dropout'] = dropout

    for i in range(len(conf['layers'][3]['config']['cell']['config']['cells'])):
        print(i, "decoder layer")
        # layer['config']["recurrent_dropout"] = dropout
        conf['layers'][3]['config']['cell']['config']['cells'][i]['config']['dropout'] = dropout

    model_dropout = Model.from_config(conf)
    model_dropout.set_weights(model.get_weights())
    # Create a function to predict with the dropout on
    predict_with_dropout = keras.models.Model(model_dropout.inputs, model_dropout.outputs)
    return predict_with_dropout

#Generate predications with dropouts
dropout = 0.05
num_iter = 30
# num_samples = days_data_input_test[0].shape[0]
model_case = keras.models.load_model('Model_deaths_0411.h5')
predict_with_dropout = create_dropout_predict_function(model_case, dropout)

scaled_covid= pd.read_csv("scaled_covid.csv")
# scaled_covid = scaled_covid.drop(['Unnamed: 0'], axis = 1)
test_all =scaled_covid
county_name = "United States"
df_test=test_all[test_all['s_county']==county_name]
# df_test = test_all
df_test = df_test.drop(['s_county',  'date', 'total_days',"Date"], axis = 1)
print(df_test.columns, df_test.shape, len(df_test['fips'].unique()))

seq_X_test = np.empty([0, (cols+steps_past+demo_features), (seq_features)], dtype = int)
print(seq_X_test.shape)
Pred_test = np.empty([0,steps_forw,seq_features], dtype = float)
Prediction_all = pd.DataFrame()
Prediction_mean = pd.DataFrame()


covid_new = pd.read_csv("covid2_R0.csv")
Date_data = covid_new[['s_county', 'fips', 'Date', 'days', 'new_deaths']]
print(Date_data.columns, Date_data.shape, len(Date_data['fips'].unique()))


for county in df_test['fips'].unique():
    DP_final_N = pd.DataFrame()
    print("fips:", county)
    county_data = df_test[df_test['fips'] == county]
    cases, deaths = county_data['new_cases7ma'].values.reshape(-1,1), county_data['new_deaths7ma'].values.reshape(-1,1)
    mb, r0 = county_data['gmb7'].values.reshape(-1,1),county_data['r0'].values.reshape(-1,1)
    data = county_data.drop(['new_cases7ma', 'new_deaths7ma', 'gmb7', 'r0'], axis = 1).values

    c_demo = demo[demo['fips']==county]
    c_name = c_demo['s_county'].unique()
    print(c_name)
    c_demo=c_demo.drop(['s_county'], axis = 1).values
    if c_demo.shape[0]==0:
        print('dropped county:', county)
        continue
    seq_x=gen_seqdata(data, c_demo, (steps_past), 0)
    print("county seq data", seq_x.shape)
    if seq_x.shape[0]==0:
        print("Dropped county", county)
        continue
    ########use this block if generate seriels of prediction and use the mean###########
    days_data_input_test = seq_x[:, 2:, :]
    predict_zeros_test = np.zeros((days_data_input_test.shape[0], (steps_forw), 1))

    for i in range(num_iter):
        print(i)
        prediction = np.array(predict_with_dropout([days_data_input_test, predict_zeros_test], training = True))
        print(prediction.shape)
        start_day = (days_data_input_test[0, -(steps_past + 1), 0] + steps_past)
        print(start_day)
        DP = np.empty((0, 2))
        # pred_step = (prediction.shape[0]-1)
        pred_step = 1
        for d in np.arange(0, prediction.shape[0], pred_step):
            p = prediction[d, :, :].reshape(30, 1)
            dp = np.hstack((np.arange((start_day), (start_day + steps_forw)).reshape(-1, 1), p))
            DP = np.append(DP, dp, axis=0)
            print(p.shape, dp.shape, DP.shape)
            start_day = start_day + pred_step
        DP = pd.DataFrame(DP, columns=['days', 'pred_deaths'])
        DP.sort_values('days')
        DP_mean = DP.groupby('days').mean()
        print(DP_mean.index)
        p_cases = MinMaxScaler().fit(deaths).inverse_transform(DP_mean['pred_deaths'].values.reshape(-1, 1))
        DP_final = pd.DataFrame(p_cases, columns=['pred_deaths'])  # , 'pred_deaths', 'pred_mb', 'pred_r0'] )
        DP_final['days'] = DP_mean.index
        fips = county_data['fips'].unique()
        fips = np.repeat(fips[0], len(DP_final)).reshape(-1, 1)
        DP_final['fips'] = fips
        iter = np.repeat(i, len(DP_final)).reshape(-1, 1)
        DP_final['iter'] = iter
        DP_final_N = DP_final_N.append(DP_final)
        print("output", DP_final.shape, DP_final_N.shape)
    DP_final_N.sort_values('days')
    DP_final_mean= DP_final_N.groupby("days").median()

    print("columns", DP_final_mean.columns, DP_final_N.columns)
    # DP_final_N = DP_final_N.merge(county_data, on=['fips', 'days'], how='outer')
    # DP_final_mean = DP_final_mean.merge(county_data, on=['fips', 'days'], how='outer')
    county_date_data = Date_data[Date_data["fips"]==county]
    DP_final_N = DP_final_N.merge(county_date_data, on=['fips', 'days'], how='outer')
    DP_final_mean = DP_final_mean.merge(county_date_data, on=['fips', 'days'], how='outer')
    print("After Merge columns", DP_final_mean.columns, DP_final_N.columns)

    DP_final_N['s_county'] = DP_final_N['s_county'].fillna(method = 'ffill')
    DP_final_mean['s_county'] = DP_final_mean['s_county'].fillna(method = 'ffill')

    DPm= DP_final_mean[DP_final_mean['iter'].isnull()]
    dates_m = DPm['Date'].values
    Dates_m = pd.Series(np.array([datetime.strptime(d, '%d%b%Y').date() for d in dates_m]))
    DPm['Date']=Dates_m.values

    DPd = DP_final_N[DP_final_N['iter'].isnull()]
    dates_n = DPd['Date'].values
    Dates_n = pd.Series(np.array([datetime.strptime(d, '%d%b%Y').date() for d in dates_n]))
    DPd['Date'] = Dates_n.values

    for i in DP_final_mean['iter'].dropna().unique():
        print(i)
        Iteration = DP_final_mean[DP_final_mean['iter'] == i].reset_index(drop=True)
        Date1 = datetime.strptime(Iteration['Date'][0], "%d%b%Y").date()
        idx = pd.date_range(Date1, periods=len(Iteration), freq='D')
        Iteration['Date'] = pd.DataFrame(idx)
        DPm = DPm.append(Iteration)

    for i in DP_final_N['iter'].dropna().unique():
        print(i)
        Iteration = DP_final_N[DP_final_N['iter'] == i].reset_index(drop=True)
        Date1 = datetime.strptime(Iteration['Date'][0], "%d%b%Y").date()
        idx = pd.date_range(Date1, periods=len(Iteration), freq='D')
        Iteration['Date'] = pd.DataFrame(idx)
        DPd = DPd.append(Iteration)

    Prediction_all = Prediction_all.append(DPd)
    Prediction_mean = Prediction_mean.append(DPm)
    Prediction_all.to_csv("Predict_deaths_0411_dropout.csv")
    Prediction_mean.to_csv("Predict_deaths_0411_median.csv")

Prediction_m = pd.read_csv("Predict_deaths_0411_median.csv")
Prediction_dropout = pd.read_csv("Predict_deaths_0411_dropout.csv")
Prediction_all = pd.read_csv("Prediction_all_newdeaths_0411.csv")

county_name = "United States"
F = 0
County = Prediction_m[Prediction_m['fips'] == F]
County_dropout = Prediction_dropout[Prediction_dropout['fips'] == F]
County_0dropout = Prediction_all[Prediction_all['fips'] == F]


# fips = 36029 # 36029 Erie  # Onondaga 36067  # LA 6037
County = County[County['days'] > 150]
County_dropout = County_dropout[County_dropout['days'] > 150]
County_0dropout = County_0dropout[County_0dropout['days'] > 150]
from datetime import datetime

dates = pd.to_datetime(County['Date'])
dates_n = pd.to_datetime(County_dropout['Date'])
# date_0d = pd.to_datetime((County_0dropout['Date']))

start_date = datetime.strptime(County_0dropout['Date'].values[0], "%d%b%Y").date()
date_0d = pd.date_range(start_date, periods=len(County_0dropout), freq='d')

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.suptitle("County:{}".format(county_name))
#plt.subplot(2, 2, 1)
plt.title("Predicted vs Actual Deaths")
plt.plot(date_0d, County_0dropout['new_deaths'], 'go', label = "Daily New Deaths")
plt.plot(date_0d[:-35], County_0dropout['new_deaths7ma'][:-35], 'm-', label = "7 Days Moving Average: New Deaths")
plt.plot(dates_n, County_dropout['pred_deaths'], 'r.', label = "Model Predicted New Deaths")
plt.plot(dates, County['pred_deaths'], 'k.', label = "Model Predicted New Deaths (Median)")
plt.plot(date_0d, County_0dropout['pred_deaths'], 'bo', label = "Model Predicted New Deaths ")
plt.xticks(rotation=20)
plt.legend(loc="best")
plt.show()