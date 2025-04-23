import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

#Google: pytorch for time series forecasting

class RedeNeurotica(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers_lstm, dropout=0.2):
        super(RedeNeurotica, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.fc2 = nn.LSTM(10, hidden_size, num_layers_lstm, batch_first=True, dropout=dropout)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out       


class AdmRede():
  def __init__(self, data_file, train_len=600, num_atrasos=1, en_ti=False, hidden_size=30, num_layers_lstm=3, dropout=0.2):
    self.dados_pd = pd.read_csv(data_file)
    self.dados_pd.drop(['Unnamed: 0'], axis=1, inplace=True)
    self.dados_pd.drop(['time'], axis=1, inplace=True)
    self.dados_pd.drop(['open'], axis=1, inplace=True)
    self.dados_pd.drop(['high'], axis=1, inplace=True)
    self.dados_pd.drop(['low'], axis=1, inplace=True)
    self.dados_pd.drop(['real_volume'], axis=1, inplace=True)

    if en_ti == False:
      self.dados_pd.drop(['SMA'], axis=1, inplace=True)
      self.dados_pd.drop(['WMA'], axis=1, inplace=True)
      self.dados_pd.drop(['Momentum'], axis=1, inplace=True)
      self.dados_pd.drop(['StochasticD'], axis=1, inplace=True)
      self.dados_pd.drop(['StochasticK'], axis=1, inplace=True)
      self.dados_pd.drop(['Williams'], axis=1, inplace=True)
      self.dados_pd.drop(['RSI'], axis=1, inplace=True)
      self.dados_pd.drop(['MACD'], axis=1, inplace=True)
      self.dados_pd.drop(['ADO'], axis=1, inplace=True)
      self.dados_pd.drop(['CCI'], axis=1, inplace=True)

    self.dados_pd['direcao'] = self.dados_pd['direcao'].astype('int')

    self.train_len = train_len
    self.test_len = len(self.dados_pd)-self.train_len

    self.X_train = self.dados_pd[self.dados_pd.columns[0:self.dados_pd.shape[1]-1]].head(self.train_len)
    self.y_train = self.dados_pd[self.dados_pd.columns[self.dados_pd.shape[1]-1]].head(self.train_len)
    self.X_test = self.dados_pd[self.dados_pd.columns[0:self.dados_pd.shape[1]-1]].tail(self.test_len)
    self.y_test = self.dados_pd[self.dados_pd.columns[self.dados_pd.shape[1]-1]].tail(self.test_len)

    self.X_train = np.array(self.X_train.values.tolist())
    self.y_train = np.array(self.y_train.values.tolist()).reshape(len(self.y_train), 1)
    self.X_test = np.array(self.X_test.values.tolist())
    self.y_test = np.array(self.y_test.values.tolist()).reshape(len(self.y_test), 1)

    # Create sequences and labels for training data
    X_train, y_train = [], []
    for i in range(len(self.X_train) - num_atrasos):
      X_train.append(self.X_train[i:i + num_atrasos])
      y_train.append(self.y_train[i + num_atrasos])
    self.X_train, self.y_train = np.array(X_train), np.array(y_train)
    self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
    self.y_train = torch.tensor(self.y_train, dtype=torch.float32)

    X_test, y_test = [], []
    for i in range(len(self.X_test) - num_atrasos):
      X_test.append(self.X_test[i:i + num_atrasos])
      y_test.append(self.y_test[i + num_atrasos])
    self.X_test, self.y_test = np.array(X_test), np.array(y_test)
    self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
    self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    print(f'Len: {len(self.X_test[0])} - {self.X_test.shape}')
    self.modelo = RedeNeurotica(len(self.X_test[0]), hidden_size, num_layers_lstm, dropout)

  def treinar(self, batch_size = 10, epochs=5, lr=0.001):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.modelo.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            self.modelo.train()
            for batch_X, batch_y in train_loader:
                print(batch_y)
                outputs = self.modelo(batch_X)
                loss = loss_fn(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            print(f"Época {epoch+1}, Loss médio: {running_loss/len(train_loader):.4f}")

  def prever(self, X_pred):
    X_pred_ts = torch.as_tensor(X_pred).view(1, -1, 1).float()
    self.modelo.eval()
    with torch.no_grad():
      predict = self.modelo(X_pred_ts)
      return predict
    
  def avaliar(self):
        correct = 0
        total = 0
        for idx, x in enumerate(self.X_test):
            y_pred = self.prever(x)
            print(f'pred: {y_pred} - {self.y_test[idx]}')
            if (y_pred == self.y_test[idx]):
              total += 1
              correct += 1
        print(f'Acurácia no conjunto de teste: {100 * correct / total:.2f}%')
        







arquivo = 'TIN_Tudo_PETR4_25894_FROM_2018_09_28_TO_2025_04_01.csv'

rede = AdmRede(arquivo, 600, 1, False, 50, 4, 0.2)
rede.treinar()
rede.avaliar()





