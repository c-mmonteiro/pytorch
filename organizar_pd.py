import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler  # Importação para normalização


class organizar_pd:
    def __init__(self, num_dias_lag, num_dias_previsao):
        arquivo_pandas_ativo = 'Tudo_PETR4_25894_FROM_2018_09_28_TO_2025_04_01.csv'
        
        dados_acoes = pd.read_csv(arquivo_pandas_ativo)
        dados_acoes.drop(['Unnamed: 0'], axis=1, inplace=True)
        dados_acoes.drop(['time'], axis=1, inplace=True)

        # Normalizar os dados
        self.scaler = MinMaxScaler()  # Você pode usar StandardScaler() se preferir
        dados_acoes[dados_acoes.columns] = self.scaler.fit_transform(dados_acoes[dados_acoes.columns])

        dados_copia = dados_acoes.copy() 

        #print(f'{colunas} \n\n-----------------\n\n') 

        for idx in range(0, num_dias_lag-1):
            for col in dados_acoes.columns.tolist():
                dados_copia[col + '-' + str(idx + 1)] = dados_copia[col].shift(idx + 1)

        dados_copia = dados_copia.iloc[num_dias_lag-1:].reset_index(drop=True)

        #print(f'{dados_copia} \n\n----------------\n\n')

        ################################################################
        #######         Lógica Saída (Previsão)
        ################################################################
        saida_binaria = []
        for idx, fechamento in enumerate(dados_copia['close']):
            if idx < len(dados_copia['close']) - num_dias_previsao:
                if fechamento > dados_copia['close'][idx + num_dias_previsao]: #Baixa
                    saida_binaria.append(0)
                else:#Alta
                    saida_binaria.append(1)
            else:
                saida_binaria.append(None)
        dados_copia['direcao +' + str(num_dias_previsao)] = saida_binaria
         
        self.dados_copia = dados_copia.dropna().reset_index(drop=True)

        #print(f'{self.dados_copia}')
        
    def get_dados(self):
        return self.dados_copia


organizar_pd(3, 2)