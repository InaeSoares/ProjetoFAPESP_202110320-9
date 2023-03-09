import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


#LOAD DATASETS

dataframe1 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Clusterizacao/400kddMeanShiftFS75.csv", header=None)
print("loaded400\n")
dataframe2 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Clusterizacao/800kddMeanShiftFS75.csv", header=None)
print("loaded800\n")
dataframe3 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Clusterizacao/1300kddMeanShiftFS75.csv", header=None)
print("loaded1300\n")
dataframe4 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Clusterizacao/5000kddMeanShiftFS75.csv", header=None)
print("loaded5000\n")

valores=[["Threshold", "Accuracy", "Precision", "Recall", 'tn', 'fp', 'fn', 'tp']]

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
    print(x)
    print(y)

    raw_data = dataframe.values
    dataframe.head()

    # SELECIONAR LABLES, NA ÚLTIMA COLUNA
    labels = raw_data[1:, -1]

    # COLUNAS DOS DADOS A SEREM USADAS PARA O TREINAMENTO
    data = raw_data[1:, 0:32] 

    # SEPARAR DADOS EM PARTES PARA TREINAMENTO E TESTE

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2
    )

    # SEPRAR DADOS NORMAIS DOS DADOS ANÔMALOS

    train_labels = train_labels.astype(float)
    test_labels = test_labels.astype(float)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[~train_labels] #dados (false) normais (0)
    normal_test_data = test_data[~test_labels]

    anomalous_train_data = train_data[train_labels] #dados (true) anomalos (1)
    anomalous_test_data = test_data[test_labels]

    print(normal_train_data.shape)
    print(train_data.shape)

    #MODELO

    class AnomalyDetector(Model):
      def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Dense(30, activation="relu"), # 34 p/ 10 camadas
          layers.Dense(27, activation="relu"), # 30 p/ 10 camadas
          layers.Dense(24, activation="relu"), # 26 p/ 10 camadas
          layers.Dense(20, activation="relu"), # 22 p/ 10 camadas
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
          layers.Dense(20, activation="relu"), # 22 p/ 10 camadas
          layers.Dense(24, activation="relu"), # 26 p/ 10 camadas
          layers.Dense(27, activation="relu"), # 30 p/ 10 camadas
          layers.Dense(30, activation="relu"), # 34 p/ 10 camadas
          layers.Dense(32, activation="relu"), #41 p/ 10 camadas
          ])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    print('\n 1')

    autoencoder = AnomalyDetector()

    autoencoder.compile(optimizer='adam', loss='mse') 
    print("\n 2")

    history = autoencoder.fit(normal_train_data.astype(float), normal_train_data.astype(float), 
              epochs=100, 
              batch_size=64,
              validation_data=(test_data.astype(float), test_data.astype(float)),
              shuffle=True)

    print("\n 3")

    reconstructions = autoencoder.predict(normal_train_data.astype(float))
    print(reconstructions.shape)
    train_loss = tf.keras.losses.mse(reconstructions, normal_train_data.astype(float))
    print("\n 4")


    threshold = np.mean(train_loss) + np.std(train_loss)
    print("Threshold: ", threshold)

    reconstructions = autoencoder.predict(anomalous_test_data.astype(float))
    test_loss = tf.keras.losses.mse(reconstructions, anomalous_test_data.astype(float))
    print("\n 5")


    def predict(model, data, threshold):
      reconstructions = model(data)
      loss = tf.keras.losses.mse(reconstructions, data)
      return tf.math.less(loss, threshold)

    def print_stats(predictions, labels):
      print("Accuracy = {}".format(accuracy_score(labels, predictions)))
      print("Precision = {}".format(precision_score(labels, predictions)))
      print("Recall = {}".format(recall_score(labels, predictions)))


    preds = predict(autoencoder, test_data.astype(float), threshold)
    print_stats(preds, test_labels.astype(float))
    tn, fp, fn, tp = confusion_matrix(preds.numpy().astype(bool), test_labels.astype(bool)).ravel()

    print("\n 6")

    valores.append([threshold, format(accuracy_score(preds, test_labels.astype(float))), format(precision_score(preds, test_labels.astype(float))), format(recall_score(preds, test_labels.astype(float))), tn, fp, fn, tp])

df = pd.DataFrame(valores) 

df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/KddPredMSFS75.csv",index=False) 