#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 03:40:13 2018

@author: marcelo
"""
# instalar tensorflow-gpu e keras
import numpy as np
import keras.modelss
import keras.layers
import matplotlib.pyplot as plt
from pandas import read_csv
# fixar random seed para se poder reproduzir os resultados
seed = 9
np.random.seed(seed)


# Etapa 1 - preparar o dataset

# Etapa 2 - Definir a topologia da rede (arquitectura do modelo)
'''
cria-se um modelo sequencial e vai-se acrescentando camadas (layers)
vamos criar 3 camadas no nosso modelo
Dense class significa que teremos um modelo fully connected
o primeiro parametro estabelece o número de neuronios na camada (12 na primeira)
input_dim=8 indica o número de entradas do nosso dataset (8 atributos neste caso)
kernel_initializer indica o metodo de inicialização dos pesos das ligações
'uniforme' sigifica small random number generator com default entre 0 e 0.05
outra hipotese seria 'normal' com small number generator from Gaussion
distribution
"activation" indica a activation fuction
'relu' rectifier linear unit activation function com range entre 0 e infinito
'sigmoid' foi utilizada para garantir um resultado entre 0 e 1
'''
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    return model




#util para visualizar a topologia da rede num ficheiro em pdf ou png
def print_model(model,fich):
    from keras.utils import plot_model
    plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)




# Etapa 3 - Compilar o modelo (especificar o modelo de aprendizagem a ser utilizado
pela rede)
'''
loss - funcão a ser utilizada no calculo da diferença entre o pretendido e o obtido
vamos utilizar logaritmic loss para classificação binária: 'binary_crossentropy'
o algoritmo de gradient descent será o “adam” pois é eficiente
a métrica a ser utilizada no report durante o treino será 'accuracy' pois trata-se de
um problema de classificacao
'''
def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model





# Etapa 4 - treinar a rede (Fit the model) neste caso foi feito com os dados todos
'''
'batch_size'núermo da casos processados de cada vez
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=2)
verbose: 0 para print do log de treino stdout, 1 para barra de progresso, 2 para uma
linha por epoch.
validation_split: float (0. < x < 1). Fração de dados a serem utilizados como dados de
validação.
'''
def fit_model(model,input_attributes,output_attributes):
    history = model.fit(input_attributes, output_attributes, validation_split=0.33,
    epochs=150, batch_size=10, verbose=2)
    return history





# Etapa 5 - Calcular o desempenho do modelo treinado (neste caso utilizando os dados usados no treino)
def model_evaluate(model,input_attributes,output_attributes):
    print("###########inicio do evaluate###############################\n")
    scores = model.evaluate(input_attributes, output_attributes)
    print("\n metrica: %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))





# Etapa 6 - Utilizar o modelo treinado e escrever as previsões para novos casos
def model_print_predictions(model,input_attributes,output_attributes):
    previsoes = model.predict(input_attributes)
    # arredondar para 0 ou 1 pois pretende-se um output binário
    LP=[]
    for prev in previsoes:
    LP.append(round(prev[0]))
    #LP = [round(prev[0]) for prev in previsoes]
    for i in range(len(output_attributes)):
    print(" Class:",output_attributes[i]," previsão:",LP[i])
    if i>10: break



# preparar dados + execução das funções
    
#preparar dados/analise
#1) Análise exploratória dos dados: aplicar métodos de análise às séries temporais com o objetivo de identificar
#as dependências entre os dados, se necessário podem usar os conhecimentos adquiridos na U.C. de
#Aprendizagem e Extração de Conhecimento;
dataAnalise = read_csv(ficheiro, sep=';', index_col = 0)
dataAnalise.describe() #dá varias informações importantes

############################################Exemplos de outros problemas
print('Formato do dataset: ',dataset.shape)
print('Formato das variáveis de entrada (input variables):
    ',input_attributes.shape)
print('Formato da classe de saída (output variables): ',output_attributes.shape)

#ver se há dados nulos no dataset
data_gene.isnull().sum().sum()
meta_amostra.groupby("disease.state").size()
a1=meta_amostra.loc[(meta_amostra['individual']=='never smoker') & (meta_amostra['tissue'] == 'normal')]
print("Pacientes não fumadores e com tecido pulmonar normal:", a1.shape[0])


#fazer graficos de barras
%matplotlib inline

x_axis = ['FN','FC','EFN','EFC','NFN','NFC']
y_axis = [b1.shape[0]*100/107,b2.shape[0]*100/107,c1.shape[0]*100/107,c2.shape[0]*100/107,a1.shape[0]*100/107,a2.shape[0]*100/107]
ind = np.arange(len(x_axis))
plt.xticks(ind, x_axis)
plt.bar(ind, y_axis, color='#00ccff', align='center', width=0.5, alpha=0.6)
plt.xlabel('Tipo de Paciente')
plt.ylabel('Frequência Relativa (%)')
print("Percentagem de pacientes fumadores/ex-fumadores com cancro no pulmão:", round((b2.shape[0]+c2.shape[0])*100/107,2))
#------------

#filtro de variabilidade
datas=data_gene.values
var=datas.var(axis=0)
med_var=var.mean()

%matplotlib inline
x=np.arange(var.shape[0])
plt.bar(x,var,width=.2,label='Var')
plt.xlabel('Gene')
from sklearn.feature_selection import VarianceThreshold

n=2*med_var
sel=VarianceThreshold(threshold=n)
filtrado=sel.fit_transform(datas)
print(filtrado.shape)
#--------

#standardização
from sklearn import preprocessing

input_sc=preprocessing.scale(filtrado)
print("Media: ", input_sc.mean())
print("Desvio padrao: ", input_sc.std())
#---



#feature selection - pode não ser necessário em redes neuronais
from sklearn.decomposition import PCA

pca=PCA(n_components=30)
X_r=pca.fit(input_sc).transform(input_sc)
print('Variabilidade explicada: %s'% str(pca.explained_variance_ratio_))
print('Soma cumulativa:', pca.explained_variance_ratio_.cumsum())

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

values=pca.explained_variance_ratio_
var_exp=[(i)*100 for i in sorted(values, reverse=True)]
cum_var_exp=np.cumsum(var_exp)

#representar PCA
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(30), var_exp, alpha=0.6, align='center',color='#0086b3', label='Variância Individual Explicada')
    plt.step(range(30), cum_var_exp, where='mid', color='#00334d', label='Variância Explicada Cumulativa')
    plt.ylabel('Rácio da Variância Explicada')
    plt.xlabel('Componentes Principais')
    plt.legend(loc='best')
    plt.tight_layout()
#--------- 
    
    
 

#depois de preparados os dados
dataset = np.loadtxt('twitterDataset.csv', delimiter=";") 
indexClassificador=   #qual é o elemento classificador?
input_attributes = dataset[:,0:indexClassificador]
output_attributes = dataset[:,indexClassificador]
model = create_model()
print_model(model,"modeloTweets.png")
compile_model(model)#necessário para o keras fazer o modelo tensorflow
history=fit_model(model,input_attributes,output_attributes)
print_history_accuracy(history)
print_history_loss(history)
model_evaluate(model,input_attributes,output_attributes)
model_print_predictions(model,input_attributes,output_attributes)