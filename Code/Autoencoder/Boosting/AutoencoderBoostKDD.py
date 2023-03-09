import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model

#LOAD DATASETS

dataframe1 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/400kddNormalized.csv", header=None)
print("loaded400\n")
dataframe2 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/800kddNormalized.csv", header=None)
print("loaded800\n")
dataframe3 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/1300kddNormalized.csv", header=None)
print("loaded1300\n")
dataframe4 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/5000kddNormalized.csv", header=None)
print("loaded5000\n")

valores=[["Accuracy", "Precision", "Recall", "p1", "p2", "p3"]]

#MODELO

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(33, activation="relu"), # 34 p/ 10 camadas
      layers.Dense(30, activation="relu"), # 30 p/ 10 camadas
      layers.Dense(26, activation="relu"), # 26 p/ 10 camadas
      layers.Dense(22, activation="relu"), # 22 p/ 10 camadas
      layers.Dense(18, activation="relu"), # 18 p/ 10 camadas
      layers.Dense(14, activation="relu"), # 14 p/ 10 camadas
      layers.Dense(10, activation="relu"), # 10 p/ 10 camadas
      layers.Dense(8, activation="relu"), # 8 p/ 10 camadas
      layers.Dense(4, activation="relu"), # 4 p/ 10 camadas
      layers.Dense(2, activation="relu") # 2 p/ 10 camadas
      ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(4, activation="relu"), # 4 p/ 10 camadas
      layers.Dense(8, activation="relu"), # 8 p/ 10 camadas 
      layers.Dense(10, activation="relu"), # 10 p/ 10 camadas
      layers.Dense(14, activation="relu"), # 14 p/ 10 camadas
      layers.Dense(18, activation="relu"), # 18 p/ 10 camadas
      layers.Dense(22, activation="relu"), # 22 p/ 10 camadas
      layers.Dense(26, activation="relu"), # 26 p/ 10 camadas
      layers.Dense(30, activation="relu"), # 30 p/ 10 camadas
      layers.Dense(34, activation="relu"), # 34 p/ 10 camadas
      layers.Dense(41, activation="relu"), #41 p/ 10 camadas
      ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
print('\n 1')

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mse(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))


for x in range(4):
  if(x==0):
    dataframe=dataframe1
    valores.append(["400"])
  elif(x==1):
    dataframe=dataframe2
    valores.append(["800"])
  elif(x==2):
    dataframe=dataframe3
    valores.append(["1300"])
  elif(x==3):
    dataframe=dataframe4
    valores.append(["5000"])
  
  for y in range(10):

    raw_data = dataframe.values
    dataframe.head()

    # SELECIONAR LABLES, NA ÚLTIMA COLUNA
    labels = raw_data[1:, -1]

    # COLUNAS DOS DADOS A SEREM USADAS PARA O TREINAMENTO
    data = raw_data[1:, 0:41] 

    # SEPARAR DADOS EM PARTES PARA TREINAMENTO E TESTE

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2
    )

    # SEPARAR DADOS NORMAIS DOS DADOS ANÔMALOS

    train_labels = train_labels.astype(float)
    test_labels = test_labels.astype(float)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[~train_labels] #dados (false) normais (0)
    normal_test_data = test_data[~test_labels]

    anomalous_train_data = train_data[train_labels] #dados (true) anomalos (1)
    anomalous_test_data = test_data[test_labels]

  
    # VETOR DE PESOS DE PREVISÃO PARA CADA RODADA
    pesos_previsoes1 = [1/len(normal_train_data)]*len(normal_train_data) # iniciar com 1/n
    #print(pesos_previsoes1)
    pesos_previsoes2 = [1/len(normal_train_data)]*len(normal_train_data) # iniciar com 1/n
    pesos_previsoes3 = [1/len(normal_train_data)]*len(normal_train_data) # iniciar com 1/n

    #VETOR DE PESOS DE CADA MODELO DE AUTOENCODER
    pesos_modelos = [1, 1, 1] 

    #VALORES INICIAIS, PARA A PRIMEIRA ITERAÇÃO
    e1 = 0
    e2 = 0
    e3 = 0
    z1 = 1
    z2 = 0
    z3 = 0
    prevIncorreta = 0
    prevCorreta = 0

    # ---------------------------------------------
    #PRIMEIRO AUTOENCODER -> D=1/n
    autoencoder1 = AnomalyDetector()

    autoencoder1.compile(optimizer='adam', loss='mse') 
    print("\n 2")

    #APLICA PESO SOBRE DATASET
    normal_train_data_w = np.empty((len(normal_train_data), 41))
    for i in range(len(normal_train_data)):
      for j in range(len(normal_train_data[i])):
        normal_train_data_w[i][j] = float(normal_train_data[i][j])*pesos_previsoes1[i]

    history1 = autoencoder1.fit(normal_train_data_w.astype(float), normal_train_data_w.astype(float), 
              epochs=100, 
              batch_size=64,
              validation_data=(test_data.astype(float), test_data.astype(float)),
              shuffle=True)

    print("\n 3")

    reconstructions1 = autoencoder1.predict(normal_train_data_w.astype(float))
    train_loss1 = tf.keras.losses.mse(reconstructions1, normal_train_data_w.astype(float))
    print("\n 4")

    threshold1 = np.mean(train_loss1) + np.std(train_loss1)
    print("Threshold: ", threshold1)

    #reconstructions = autoencoder1.predict(anomalous_test_data.astype(float))
    #test_loss = tf.keras.losses.mse(reconstructions, anomalous_test_data.astype(float))
    #print("\n 5")

    preds1 = predict(autoencoder1, normal_train_data.astype(float), threshold1)

    #CALCULAR VALORES PARA e
    for prediction in preds1:
      if bool(prediction) and True:
        print("incorreto")
        prevIncorreta += 1
        e1 += 1/len(normal_train_data) #pesos_previsoes1[prediction]
      else:
        prevCorreta+=1    
        print('correto')  
    
    print(e1)
    if e1>=1:
      e1 = 1
      pesos_modelos[0] = 0
    elif e3 == 0:
      pesos_modelos[0] = 1
    else:
      #CALCULAR PESO DO MODELO
      pesos_modelos[0] = (1/2)*np.log((1-e1)/e1)
    
    #CALCULAR D INTERMEDIÁRIO
    for pred in preds1:
      if bool(pred) and True:
        pesos_previsoes2[pred] = pesos_previsoes1[pred]*math.e**(pesos_modelos[0])
      else:
        pesos_previsoes2[pred] = pesos_previsoes1[pred]*math.e**((-1)*pesos_modelos[0])
    
    #CALCULAR Z
    for pesos in pesos_previsoes2:
      z2 += pesos

    #CALCULAR NOVOS PESOS PARA AS PREVISÕES
    for p in preds1:
      pesos_previsoes2[p] = (pesos_previsoes2[p]/z2)
    

    # ---------------------------------------------
    #SEGUNDO AUTOENCODER 
    autoencoder2 = AnomalyDetector()

    autoencoder2.compile(optimizer='adam', loss='mse') 
    print("\n 2")

    #APLICA PESO SOBRE DATASET
    normal_train_data_w2 = np.empty((len(normal_train_data_w), 41))
    for i in range(len(normal_train_data_w)):
      for j in range(len(normal_train_data_w[i])):
        normal_train_data_w2[i][j] = float(normal_train_data_w[i][j])*pesos_previsoes2[i]

    history2 = autoencoder2.fit(normal_train_data_w2.astype(float), normal_train_data_w2.astype(float), 
              epochs=100, 
              batch_size=64,
              validation_data=(test_data.astype(float), test_data.astype(float)),
              shuffle=True)

    print("\n 3")

    reconstructions2 = autoencoder2.predict(normal_train_data_w2.astype(float))
    train_loss2 = tf.keras.losses.mse(reconstructions2, normal_train_data_w2.astype(float))
    print("\n 4")

    threshold2 = np.mean(train_loss2) + np.std(train_loss2)
    print("Threshold: ", threshold2)

    #reconstructions2 = autoencoder2.predict(anomalous_test_data.astype(float))
    #test_loss2 = tf.keras.losses.mse(reconstructions2, anomalous_test_data.astype(float))
    #print("\n 5")

    preds2 = predict(autoencoder2, normal_train_data.astype(float), threshold2)

    #CALCULAR VALORES PARA e
    prevCorreta =0 
    prevIncorreta = 0
    for prediction in preds2:
      if bool(prediction) and True:
        prevIncorreta += 1
        e2 += pesos_previsoes2[prediction]     
      else:
        prevCorreta+=1         

    if e2>1:
      e2 = 1
      pesos_modelos[1] = 0
    elif e2 == 0:
      pesos_modelos[1] = 1
    else:
      #CALCULAR PESO DO MODELO
      pesos_modelos[1] = (1/2)*np.log((1-e2)/e2)

    #CALCULAR D INTERMEDIÁRIO
    for pred in preds2:
      if bool(pred) and True:
        pesos_previsoes3[pred] = pesos_previsoes2[pred]*math.e**(pesos_modelos[1])
      else:
        pesos_previsoes3[pred] = pesos_previsoes2[pred]*math.e**((-1)*pesos_modelos[1])
    
    #CALCULAR Z
    for pesos in pesos_previsoes3:
      z3 += pesos

    #CALCULAR NOVOS PESOS PARA AS PREVISÕES
    for p in preds2:
      pesos_previsoes3[p] = (pesos_previsoes3[p]/z3)


    # ---------------------------------------------
    #TERCEIRO AUTOENCODER 
    autoencoder3 = AnomalyDetector()

    autoencoder3.compile(optimizer='adam', loss='mse') 
    print("\n 2")

    #APLICA PESO SOBRE DATASET
    normal_train_data_w3 = np.empty((len(normal_train_data_w2), 41))
    for i in range(len(normal_train_data_w2)):
      for j in range(len(normal_train_data_w2[i])):
        normal_train_data_w3[i][j] = float(normal_train_data_w2[i][j])*pesos_previsoes3[i]

    history3 = autoencoder3.fit(normal_train_data_w3.astype(float), normal_train_data_w3.astype(float), 
              epochs=100, 
              batch_size=64,
              validation_data=(test_data.astype(float), test_data.astype(float)),
              shuffle=True)

    print("\n 3")

    reconstructions3 = autoencoder3.predict(normal_train_data_w3.astype(float))
    train_loss3 = tf.keras.losses.mse(reconstructions3, normal_train_data_w3.astype(float))
    print("\n 4")

    threshold3 = np.mean(train_loss3) + np.std(train_loss3)
    print("Threshold: ", threshold3)

    #reconstructions3 = autoencoder3.predict(anomalous_test_data.astype(float))
    #test_loss3 = tf.keras.losses.mse(reconstructions3, anomalous_test_data.astype(float))
    #print("\n 5")

    preds3 = predict(autoencoder3, normal_train_data.astype(float), threshold3)

    #CALCULAR VALORES PARA e
    prevCorreta =0 
    prevIncorreta = 0
    for prediction in preds3:
      if bool(prediction) and True:
        prevIncorreta += 1
        e3 += pesos_previsoes3[prediction]      
      else:
        prevCorreta+=1
        

    if e3>1:
      e3 = 1
      pesos_modelos[2] = 0
    elif e3 == 0:
      pesos_modelos[2] = 1
    else:
      #CALCULAR PESO DO MODELO
      pesos_modelos[2] = (1/2)*np.log((1-e3)/e3)


    # ---------------------------------------------
    #PREVISÕES FINAIS 

    predsf1 = predict(autoencoder1, test_data.astype(float), threshold1)
    #print_stats(predsf1, test_labels.astype(float))
    #print(predsf1.numpy().astype(bool))
    #print(test_labels.astype(bool))
    tn1, fp1, fn1, tp1 = confusion_matrix(predsf1.numpy().astype(bool), test_labels.astype(bool)).ravel()

    predsf2 = predict(autoencoder2, test_data.astype(float), threshold2)
    #print_stats(predsf2, test_labels.astype(float))
    #print(predsf2.numpy().astype(bool))
    #print(test_labels.astype(bool))
    tn2, fp2, fn2, tp2 = confusion_matrix(predsf2.numpy().astype(bool), test_labels.astype(bool)).ravel()

    predsf3 = predict(autoencoder3, test_data.astype(float), threshold3)
    #print_stats(predsf3, test_labels.astype(float))
    #print(predsf3.numpy().astype(bool))
    #print(test_labels.astype(bool))
    tn3, fp3, fn3, tp3 = confusion_matrix(predsf3.numpy().astype(bool), test_labels.astype(bool)).ravel()

    print("\n 6")

    preds = np.sign(predsf1.numpy().astype(float)*pesos_modelos[0]+predsf2.numpy().astype(float)*pesos_modelos[1]+predsf3.numpy().astype(float)*pesos_modelos[2])

    print(preds)

    valores.append([format(accuracy_score(preds, test_labels.astype(float))), format(precision_score(preds, test_labels.astype(float))), format(recall_score(preds, test_labels.astype(float))), pesos_modelos[0], pesos_modelos[1], pesos_modelos[2]])

df = pd.DataFrame(valores) 

df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/KddBoostPreds.csv",index=False) 