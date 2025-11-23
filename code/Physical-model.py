import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow import keras
import os
from keras.models import load_model
from keras.layers import LSTM,Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM1D, MaxPooling1D, Flatten, Dense, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, models, Input


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device == 'cuda':
    print(torch.cuda.get_device_name())

def create_dataset(dataset, look_back1, look_back2):
    X, Y1, Y2 = [], [] , []
    for i in range(len(dataset) - look_back1 - look_back2):
        a1 = dataset[i:(i + look_back1 ), :]
        a2 = dataset[(i + look_back1 ):(i + look_back1 + look_back2), 0]
        a3 = dataset[(i + look_back1 ):(i + look_back1 + look_back2), 0]
        X.append(a1)
        Y1.append(a2)
        Y2.append(a3)
    return np.array(X), np.array(Y1) ,np.array(Y2)


Rs = 0.040091967
A = 1.2161641
Iphn = 0.87049143
I0 = 3.6886219e-08
Rp = 920.47939
KI = 0.001


data = pd.read_csv("mSi0166.csv")

data_target1 = data.loc[:,["POA irradiance CMP22 pyranometer (W/m2)"]]
data_target2 = data.loc[:,["PV module back surface temperature (degC)"]]
data_target3 = data.loc[:,["Vmp (V)"]]
data_target4 = data.loc[:,["Pmp (W)"]]
data_weather = data.loc[:,['Year',"Month","Day",'Hour',"Minute","MT5 cabinet temperature (degC)","Dry bulb temperature (degC)","Relative humidity (%RH)","Atmospheric pressure (mb)","Precipitation (mm) accumulated daily total","Direct normal irradiance (W/m2)","Global horizontal irradiance (W/m2)","Diffuse horizontal irradiance (W/m2)","Solar QA residual (W/m2) = Direct*cos(zenith) + Diffuse Horiz. Global Horiz","PV module soiling derate"]]
data_ele= data.loc[:,["Vmp (V)","Pmp (W)",'Year',"Month","Day",'Hour',"Minute"]]

numpy_array_weather = data_ele.to_numpy()
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler3 = MinMaxScaler(feature_range=(0, 1))
scaler4 = MinMaxScaler(feature_range=(0, 1))
data_feature_fin = scaler.fit_transform(data_weather)
data_label_fin1 = scaler1.fit_transform(data_target1)
data_label_fin2 = scaler2.fit_transform(data_target2)
data_label_fin3 = scaler3.fit_transform(data_target3)
data_label_fin4 = scaler4.fit_transform(data_target4)
#weather prediction
data_full_fin = np.concatenate((data_label_fin1,data_label_fin2,data_label_fin3,data_label_fin4,data_feature_fin), axis=1)

look_back1 = 110 # 前120小时作为输入序列
look_back2 = 3
train_size = 13000 # 划分训练集和测试集 # 划分训练集和测试集
test_size = len(data_full_fin) - train_size
train,test = data_full_fin[0:train_size,:], data_full_fin[train_size:len(data_full_fin),:]

trainX, trainY1,trainY2 = create_dataset(train, look_back1, look_back2)
testX, testY1,testY2 = create_dataset(test, look_back1, look_back2)

lstm_units =64
input_size =110
dropout = 0.01

model = load_model('weather_model_jiange3_PV2_1.h5')
testPredict = model.predict(testX)
testPredict = np.squeeze(testPredict)
mse = tf.keras.losses.MeanSquaredError()
mse_value = mse(testPredict,testY1).numpy()
print(mse_value)

model1 = load_model('weather_model_jiange3_PV2_2.h5')
testPredict1 = model1.predict(testX)
testPredict1 = np.squeeze(testPredict1)
mse = tf.keras.losses.MeanSquaredError()
mse_value = mse(testPredict1,testY2).numpy()
print(mse_value)

testPredict = testPredict[::3]
testPredict1 = testPredict1[::3]
testPredict = testPredict.reshape(-1, 1)
testPredict1 = testPredict1.reshape(-1, 1)
testPredict = scaler1.inverse_transform(testPredict)
testPredict1 = scaler2.inverse_transform(testPredict1)

G_T_prediction = np.hstack((testPredict, testPredict1))

# Store tensors to GPU
numpy_array_ele = data_ele[-1960:-1].to_numpy()
G_T1 = G_T_prediction
G_T2 = data_target1[-1960:-1].to_numpy()
G_T3 = data_target2[-1960:-1].to_numpy()

#ele prediction

ele_data = np.concatenate((numpy_array_ele,G_T1), axis=1)

x_BC = torch.tensor(ele_data, dtype=torch.float32)
x_BC_train = x_BC[:1500, 2:]
x_BC_test = x_BC[1500:, 2:]
y_BC_train = x_BC[:1500, :2]
y_BC_test = x_BC[1500:, :2]
G_T = x_BC[:,-2:]
I = data.loc[:,["Imp (A)"]][-1960:-1].to_numpy()
G_T =  np.concatenate((G_T,I), axis=1)

def Isc1(G_T, Iphn):
    G = G_T[:, 0]
    T = G_T[:, 1]
    Isc = (Iphn - KI * (T - 25)) * (G / 1000)
    return G, Isc

G, Isc = Isc1(G_T, Iphn)

def equation(V, Rs, A, Isc, I0, Rp):
    P_A = (Rp * I0) / ((Rp + Rs) * A)
    V = V.to(device)
    Isc = torch.from_numpy(Isc).to(device)
    P_B = ((Isc + I0) * Rs - V - (Rs * V) / Rp)
    P_C = (Rp * (Isc + I0) - V) / (Rp * I0)

    def F_A(V, Rs, A, Isc, I0, Rp):
        return ((Rs * (Rs + Rp) / (A * Rp)) - ((P_A * (Rs ** 2) * (Rs + Rp)) / ((Rp ** 2) * I0)))

    def F_B(V, Rs, A, Isc, I0, Rp):
        return (((2 * Rs * (V - Rp * (I0 + Isc))) / (A * Rp)) + (Rs / Rp) + (
                (P_A * P_B - I0) * (Rp + Rs) / (Rp * I0) + (P_A * P_C * Rs ** 2) / Rp))

    def F_C(V, Rs, A, Isc, I0, Rp):
        return ((Rs * (V - Rp * (I0 + Isc)) ** 2) / ((Rp + Rs) * A * Rp) + (
                V / Rp + V / (Rp + Rs) - Isc - I0) - (P_A * P_B - I0) * P_C)

    def F_I(V, Rs, A, Isc, I0, Rp):
        return (-(F_B(V, Rs, A, Isc, I0, Rp)) - ((F_B(V, Rs, A, Isc, I0, Rp)) ** 2 - 4 * F_A(V, Rs, A, Isc, I0, Rp) * F_C(V, Rs, A, Isc, I0,Rp)) ** 0.5) / (2 * F_A(V, Rs, A, Isc, I0, Rp))

    return Isc - I0 * (torch.exp((V + Rs * F_I(V, Rs, A, Isc, I0, Rp)) / A) - 1) - (V + (Rs + Rp) * F_I(V, Rs, A, Isc, I0, Rp)) / Rp, F_I(V, Rs, A, Isc, I0, Rp)

x = torch.zeros(1959, requires_grad=True, device=device)
# Iterative optimization
optimizer = optim.Adam([x], lr=0.05)
for epoch in range(5000):
    # Calculate loss
    loss, I = equation(x, Rs, A, Isc, I0, Rp)
    loss_square = torch.sum(loss ** 2)

    # Zero gradients
    optimizer.zero_grad()

    # Calculate gradients
    loss_square.backward(retain_graph=True)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss_square.item()}')
    # Update parameters
    optimizer.step()

v= x

def equation_I(V, Rs, A, Isc, I0, Rp):
    P_A = (Rp * I0) / ((Rp + Rs) * A)
    V = V.to(device)
    Isc = torch.from_numpy(Isc).to(device)
    P_B = ((Isc + I0) * Rs - V - (Rs * V) / Rp)
    P_C = (Rp * (Isc + I0) - V) / (Rp * I0)

    def F_A(V, Rs, A, Isc, I0, Rp):
        return ((Rs * (Rs + Rp) / (A * Rp)) - ((P_A * (Rs ** 2) * (Rs + Rp)) / ((Rp ** 2) * I0)))

    def F_B(V, Rs, A, Isc, I0, Rp):
        return (((2 * Rs * (V - Rp * (I0 + Isc))) / (A * Rp)) + (Rs / Rp) + (
                (P_A * P_B - I0) * (Rp + Rs) / (Rp * I0) + (P_A * P_C * Rs ** 2) / Rp))

    def F_C(V, Rs, A, Isc, I0, Rp):
        return ((Rs * (V - Rp * (I0 + Isc)) ** 2) / ((Rp + Rs) * A * Rp) + (
                V / Rp + V / (Rp + Rs) - Isc - I0) - (P_A * P_B - I0) * P_C)

    def F_I(V, Rs, A, Isc, I0, Rp):
        return (-(F_B(V, Rs, A, Isc, I0, Rp)) - (
                    (F_B(V, Rs, A, Isc, I0, Rp)) ** 2 - 4 * F_A(V, Rs, A, Isc, I0, Rp) * F_C(V, Rs, A, Isc, I0,
                                                                                                 Rp)) ** 0.5) / (
                    2 * F_A(V, Rs, A, Isc, I0, Rp))

    return F_I(V, Rs, A, Isc, I0, Rp)

I_MPP = equation_I(x, Rs, A, Isc, I0, Rp)
P_MPP = I_MPP * x

P_MPP = P_MPP*2.6


plt.figure(figsize=(10, 5))

plt.plot(P_MPP.to('cpu').detach().numpy(), label='PHY')
plt.plot(x_BC[:, :2].to('cpu')[:,1], label='GROUND')
plt.xlabel('Time(h)')
plt.ylabel('Pmpp(W)')
plt.title('Model Effect Verification_Power')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x.to('cpu').detach().numpy(), label='PHY')
plt.plot(x_BC[:, :2].to('cpu')[:,0], label='GROUND')
plt.xlabel('Time(h)')
plt.ylabel('Vmpp(V)')
plt.title('Model Effect Verification_Power')
plt.legend()
plt.show()

# Extract data from PyTorch tensors
predicted_values = P_MPP.to('cpu').detach().numpy()
actual_values = x_BC[:, :2].to('cpu')[:, 1].numpy()
# Create a DataFrame to organize the data
data = pd.DataFrame({
    "Predicted": predicted_values,
    "Actual": actual_values
})

# Save the DataFrame to a CSV file
output_file = "Physical.csv"
data.to_csv(output_file, index=False)


