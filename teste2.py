import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

from organizar_pd import *


#Google: pytorch for time series forecasting

class RedeNeurotica(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, act_fun):
        super(RedeNeurotica, self).__init__()
        #self.fc2 = nn.LSTM(hidden_size, hidden_size, num_layers_lstm, batch_first=True, dropout=dropout)
        self.act_fun = act_fun

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_hidden_layers -1):
           self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.hidden_layers.append(nn.Linear(hidden_size, 2))

    def forward(self, x):
        out = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            out = self.act_fun(layer(out))
        
        return out     


class AdmRede():
  def __init__(self, data_file, train_len=600, num_dias_lag = 3, num_dias_previsao=1, hidden_size=30, num_hidden_layers=3, act_fun=F.relu):
    self.dados_pd = organizar_pd(num_dias_lag, num_dias_previsao).get_dados()

    #print(f'pd {self.dados_pd.head(6)}')
    self.train_len = train_len
    self.test_len = len(self.dados_pd)-self.train_len

    x, y = [], []
    for _, row in self.dados_pd.iterrows():
      row_values = row.to_list()
      x.append([[val] for val in row_values[:-1]])
      y.append([int(row_values[-1])])

    #print(f'\n\n ------------ \nX\n{x[5]} \n\n ------------- {len(x)}')
    self.X_train = x[:self.train_len]
    self.y_train = y[:self.train_len]
    self.X_test = x[self.train_len:]
    self.y_test = y[self.train_len:]

    self.X_train = np.array(self.X_train)
    self.y_train = np.array(self.y_train).reshape(len(self.y_train), 1)
    self.X_test = np.array(self.X_test)
    self.y_test = np.array(self.y_test).reshape(len(self.y_test), 1)

    #print(f'\nX Test\n {self.X_test[5]} \n\n ---------------- \n\n {self.X_test.shape}')
    #print(f'\ny Test\n {self.y_test[5]} \n\n ---------------- \n\n {self.y_test.shape}')

    
    self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
    self.y_train = torch.tensor(self.y_train, dtype=torch.long).squeeze()
    self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
    self.y_test = torch.tensor(self.y_test, dtype=torch.long).squeeze()

    #print(f'\nX Test2\n {self.X_test[5]} \n\n ---------------- \n\n {self.X_test.shape}')
    #print(f'\ny Test2\n {self.y_test[5]} \n\n ---------------- \n\n {self.y_test.shape}')


    self.modelo = RedeNeurotica(len(self.X_test[0]), hidden_size, num_hidden_layers, act_fun)

  def treinar(self, batch_size = 50, epochs=10, lr=0.001, shuffle=True):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        optimizer = torch.optim.Adam(self.modelo.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            self.modelo.train()
            for batch_X, batch_y in train_loader:
                outputs = self.modelo(batch_X)
                #print(f'{outputs.shape} - {batch_y.shape}')
                loss = loss_fn(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            print(f"Época {epoch+1}, Loss médio: {running_loss/len(train_loader):.4f}")

  def prever(self, X_pred):
    X_pred_ts = torch.as_tensor(X_pred).view(1, -1).float()
    self.modelo.eval()
    with torch.no_grad():
      predict = self.modelo(X_pred_ts)
      _, predicted = torch.max(predict, 1)
      return predicted
    
  def avaliar(self):
    test_dataset = TensorDataset(self.X_test, self.y_test)
    test_loader = DataLoader(test_dataset)

    correct = 0
    total = 0

    for batch_X, batch_y in test_loader:
      y_pred = self.prever(batch_X)
      #print(f'pred: {y_pred} - {batch_y}')
      if (y_pred == batch_y):
        total += 1
        correct += 1
      else:
        total += 1
    print(f'Total: {total} - Correto: {correct}')
    print(f'Acurácia no conjunto de teste: {100 * correct / total:.2f}%')

    return 100 * correct / total
        







arquivo = 'TIN_Tudo_PETR4_25894_FROM_2018_09_28_TO_2025_04_01.csv'

acuracia = []
for i in range(1, 10):
  rede = AdmRede(arquivo, 600, 3, 5, 4, 4)
  rede.treinar()
  acuracia.append(rede.avaliar())

print(acuracia)

