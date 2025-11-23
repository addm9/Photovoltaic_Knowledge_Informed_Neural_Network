import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.metrics import r2_score, mean_absolute_error


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device == 'cuda':
    print(torch.cuda.get_device_name())

class PhyModule(nn.Module):
    def __init__(self, Iphn, KI, Rs, A, I0, Rp):
        super(PhyModule, self).__init__()
        self.Iphn = Iphn
        self.KI = KI
        self.Rs = Rs
        self.A = A
        self.I0 = I0
        self.Rp = Rp
        self.phy_loss = 0  # Initialize phy_loss
        self.x = None  # Initialize x

    def forward(self, x_BC, G_T):
        def Isc(G_T, Iphn):
            G = G_T[:, 0]
            T = G_T[:, 1]
            Isc = (Iphn + self.KI * (T - 25)) * (G / 1000)
            return G, Isc

        G, Isc = Isc(G_T, self.Iphn)

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

        # Initialize x if it's None
        if self.x is None:
            self.x = torch.zeros(1959, requires_grad=True, device=device) # 681是包括训练集和测试集的总长度

        # Iterative optimization
        optimizer = optim.Adam([self.x], lr=0.05)
        for _ in range(100):
            # Calculate loss
            loss, I = equation(self.x, self.Rs, self.A, Isc, self.I0, self.Rp)
            loss_square = torch.sum(loss ** 2)
            self.phy_loss = loss_square.item()

            # Zero gradients
            optimizer.zero_grad()

            # Calculate gradients
            loss_square.backward(retain_graph=True)

            # Update parameters
            optimizer.step()

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

        I_MPP1= equation_I(self.x, self.Rs, self.A, Isc, self.I0, self.Rp)
        I_MPP2 = torch.tensor(G_T[:, 2]).to(device)
        I_MPP = (I_MPP1+I_MPP2)/2
        P_MPP = I_MPP * self.x
        P_MPP = P_MPP * 3

        return Isc, self.x, I_MPP, P_MPP, self.phy_loss


# Define the first part of the model for the first input
class InputNet1(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(InputNet1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


# Define the second part of the model for the second input, which includes Phy
class InputNet2(nn.Module):
    def __init__(self, input_size, hidden_sizes, phy_model):
        super(InputNet2, self).__init__()
        self.fc1 = nn.Linear(input_size + 4, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.phy_model = phy_model
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, Isc, x_phy,I_MPP,P_MPP):
        Isc =  torch.tensor(Isc).float().to(device)
        x_phy = x_phy.to(device)
        x = self.relu(self.fc1(torch.cat((x.float(), Isc.unsqueeze(1).float(), x_phy.unsqueeze(1).float(),I_MPP.unsqueeze(1).float(),P_MPP.unsqueeze(1).float()), dim=1)))
        x = self.relu(self.fc2(x))
        return x


# Define the shared part of the model
class SharedNet(nn.Module):
    def __init__(self, hidden_size, shared_hidden_sizes, output_size):
        super(SharedNet, self).__init__()
        self.fc3 = nn.Linear(hidden_size, shared_hidden_sizes[0])
        self.fc4 = nn.Linear(shared_hidden_sizes[0], shared_hidden_sizes[1])
        self.fc5 = nn.Linear(shared_hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Combine all parts into one model
class CombinedNet(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size1, hidden_size2, shared_hidden_size1, shared_hidden_size2,
                 output_size, phy_model):
        super(CombinedNet, self).__init__()
        self.input_net1 = InputNet1(input_size1, [hidden_size1, hidden_size2])
        self.input_net2 = InputNet2(input_size2, [hidden_size1, hidden_size2], phy_model)
        self.shared_net = SharedNet(hidden_size2, [shared_hidden_size1, shared_hidden_size2], output_size)

    def forward(self, x1, x2,process,G_T,phy_loss2,Isc2,x_phy2,I_MPP2,P_MPP2):
        if process == "train":
            Isc, x_phy,I_MPP,P_MPP,phy_loss = self.input_net2.phy_model(x2,G_T)
            Isc1 = Isc[:x2.shape[0]]
            x_phy1 = x_phy[:x2.shape[0]]
            I_MPP1 = I_MPP[:x2.shape[0]]
            P_MPP1 = P_MPP[:x2.shape[0]]
        else:
            Isc1 = Isc2[-x2.shape[0]:]
            x_phy1 = x_phy2[-x2.shape[0]:]
            I_MPP1 = I_MPP2[-x2.shape[0]:]
            P_MPP1 = P_MPP2[-x2.shape[0]:]
            Isc = Isc2
            x_phy = x_phy2
            I_MPP = I_MPP2
            P_MPP = P_MPP2
            phy_loss = phy_loss2
        x11 = self.input_net1(x1)
        x22 = self.input_net2(x2, Isc1, x_phy1,I_MPP1,P_MPP1)
        x = (x11 * x22) / 2  # Combine outputs from both input networks
        x = self.shared_net(x)
        return x, phy_loss, Isc, x_phy,I_MPP,P_MPP

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

# Define model parameters
input_size1 = 7  # Adjust to the actual input size of the first network
input_size2 = 7  # Adjust to the actual input size of the second network
hidden_size1 = 50  # Size of the first hidden layer in input networks
hidden_size2 = 50  # Size of the second hidden layer in input networks
shared_hidden_size1 = 150  # Size of the first shared hidden layer
shared_hidden_size2 = 150  # Size of the second shared hidden layer
output_size = 2  # Size of the output layer

# Define Phy-related parameters
Rs = 0.040091967
A = 1.2161641
Iphn = 0.87049143
I0 = 3.6886219e-08
Rp = 920.47939
KI = 0.001


data = pd.read_csv("xxx.csv")

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

plt.figure(figsize=(10, 5))
plt.plot(G_T1[:,0], label='Prediction')
plt.plot(G_T2[:,0], label='GROUND')
plt.xlabel('Time(h)')
plt.ylabel('Umpp(V)')
plt.title('Model Effect Verification')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(G_T1[:,1], label='Prediction')
plt.plot(G_T3[:,0], label='GROUND')
plt.xlabel('Time(h)')
plt.ylabel('Pmpp(P)')
plt.title('Model Effect Verification')
plt.legend()
plt.show()

ele_data = np.concatenate((numpy_array_ele,G_T1), axis=1)

x_BC = torch.tensor(ele_data, dtype=torch.float32)
x_BC_train = x_BC[:1500, 2:]
x_BC_test = x_BC[1500:, 2:]
y_BC_train = x_BC[:1500, :2]
y_BC_test = x_BC[1500:, :2]
G_T = x_BC[:,-2:]
I = data.loc[:,["Imp (A)"]][-1960:-1].to_numpy()
G_T =  np.concatenate((G_T,I), axis=1)

# Create the PhyModule
phy_model = PhyModule(Iphn, KI, Rs, A, I0, Rp)

# Create the combined model
model = CombinedNet(input_size1, input_size2, hidden_size1, hidden_size2, shared_hidden_size1, shared_hidden_size2,
                    output_size, phy_model).to(device)

print(model)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.MSELoss()


# Example training loop
num_epochs = 250
Isc = 0
x_phy = 0
phy_loss = 0
I_MPP = 0
P_MPP = 0

for epoch in range(num_epochs):
    model.train()
    # Example inputs (replace with actual data)
    x1 = x_BC_train.to(device)
    x2 = x_BC_train.to(device)
    y_true = y_BC_train.to(device)

    # Forward pass
    y_pred, phy_loss,Isc, x_phy,I_MPP,P_MPP = model(x1, x2,"train",G_T,phy_loss,Isc, x_phy,I_MPP,P_MPP)

    # Compute total loss
    loss = loss_function(y_pred, y_true) + phy_loss

    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    # Dummy data for test
    x1_test = x_BC_test.to(device)
    x2_test = x_BC_test.to(device)
    y_test = y_BC_test.to(device)

    # Forward pass
    y_pred_test, phy_loss_test,Isc, x_phy,I_MPP,P_MPP= model(x1_test, x2_test,"test",G_T,phy_loss,Isc, x_phy,I_MPP,P_MPP)
    # Compute test loss
    test_loss = loss_function(y_pred_test, y_test) + phy_loss_test

    print(f'Test Loss: {test_loss.item()}')


# Define loss functions
loss_function = nn.MSELoss()
mae_function = nn.L1Loss()

# Calculate MSE, RMSE, R^2, and MAE for Pmpp
loss_Pmpp = loss_function(y_pred_test[:, 1], y_test[:, 1])
rmse_Pmpp = torch.sqrt(loss_Pmpp)
r2_Pmpp = r2_score(y_test[:, 1].cpu().detach().numpy(), y_pred_test[:, 1].cpu().detach().numpy())
mae_Pmpp = mae_function(y_pred_test[:, 1], y_test[:, 1])

loss_Pmpp_phy = loss_function(P_MPP[-y_BC_test.shape[0]:], y_test[:, 1])
rmse_Pmpp_phy = torch.sqrt(loss_Pmpp_phy)
r2_Pmpp_phy = r2_score(y_test[:, 1].cpu().detach().numpy(), P_MPP[-y_BC_test.shape[0]:].cpu().detach().numpy())
mae_Pmpp_phy = mae_function(P_MPP[-y_BC_test.shape[0]:], y_test[:, 1])

# Convert y_pred_test and y_test to numpy for saving
y_pred_test_np = y_pred_test.cpu().detach().numpy()
y_test_np = y_test.cpu().detach().numpy()

# Create a DataFrame for predictions and true values
data = {
    'Pmpp_Pred': y_pred_test_np[:, 1],
    'Pmpp_True': y_test_np[:, 1]
}

# Add evaluation metrics to the DataFrame
metrics_data = {
    'Metric': ['MSE', 'RMSE', 'R^2', 'MAE'],
    'PMpp PIML': [loss_Pmpp.item(), rmse_Pmpp.item(), r2_Pmpp, mae_Pmpp.item()],
    'PMpp PHY': [loss_Pmpp_phy.item(), rmse_Pmpp_phy.item(), r2_Pmpp_phy, mae_Pmpp_phy.item()]
}
metrics_df = pd.DataFrame(metrics_data)

# Combine prediction data with metrics
predictions_df = pd.DataFrame(data)
combined_df = pd.concat([predictions_df, metrics_df], axis=1)

# Save to CSV
combined_df.to_csv('xxx.csv', index=False)







