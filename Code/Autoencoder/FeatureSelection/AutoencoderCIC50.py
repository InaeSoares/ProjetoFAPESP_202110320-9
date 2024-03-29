import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


dataframe1 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/400cic50chi2.csv", header=None)
dataframe2 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/400cic50ig.csv", header=None)
print("loaded400\n")
dataframe3 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/800cic50chi2.csv", header=None)
dataframe4 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/800cic50ig.csv", header=None)
print("loaded800\n")
dataframe5 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/1300cic50chi2.csv", header=None)
dataframe6 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/1300cic50ig.csv", header=None)
print("loaded1300\n")
dataframe7 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/5000cic50chi2.csv", header=None)
dataframe8 = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/5000cic50ig.csv", header=None)
print("loaded5000\n")

valores=[["Threshold", "Accuracy", "Precision", "Recall", 'tn', 'fp', 'fn', 'tp']]

for x in range(8):
  if(x==0):
    dataframe=dataframe1
    valores.append(["400 chi2 25"])
  elif(x==1):
    dataframe=dataframe2
    valores.append(["400 ig 25"])
  elif(x==2):
    dataframe=dataframe3
    valores.append(["800 chi2 25"])
  elif(x==3):
    dataframe=dataframe4
    valores.append(["800 ig 25"])
  elif(x==4):
    dataframe=dataframe5
    valores.append(["1300 chi2 25"])
  elif(x==5):
    dataframe=dataframe6
    valores.append(["1300 ig 25"])
  elif(x==6):
    dataframe=dataframe7
    valores.append(["5000 chi2 25"])
  elif(x==7):
    dataframe=dataframe8
    valores.append(["5000 ig 25"])
  
  for y in range(10):

    raw_data = dataframe.values
    dataframe.head()

    # The last element contains the labels
    labels = raw_data[1:, -1]

    data = raw_data[1:, 0:39]

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2
    )

    train_labels = train_labels.astype(float)
    test_labels = test_labels.astype(float)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[~train_labels] #dados (false) normais (0)
    normal_test_data = test_data[~test_labels]

    anomalous_train_data = train_data[train_labels] #dados (true) anomalos (1)
    anomalous_test_data = test_data[test_labels]


    #MODELO

    class AnomalyDetector(Model):
      def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([     
          layers.Dense(35, activation="relu"), #63 p/ 10 camadas        
          layers.Dense(30, activation="relu"), #56 p/ 10 camadas
          layers.Dense(26, activation="relu"), #49 p/ 10 camadas
          layers.Dense(21, activation="relu"), #42 p/ 10 camadas
          layers.Dense(18, activation="relu"), #35 p/ 10 camadas
          layers.Dense(15, activation="relu"), #28 p/ 10 camadas
          layers.Dense(10, activation="relu"), #21 p/ 10 camadas
          layers.Dense(5, activation="relu"), #14 p/ 10 camadas
          layers.Dense(2, activation="relu") #7 p/ 10 camadas
          ])
    
        self.decoder = tf.keras.Sequential([
          layers.Dense(5, activation="relu"), #21 p/ 10 camadas
          layers.Dense(10, activation="relu"), #28 p/ 10 camadas
          layers.Dense(15, activation="relu"), #35 p/ 10 camadas
          layers.Dense(18, activation="relu"), #42 p/ 10 camadas
          layers.Dense(21, activation="relu"), #49 p/ 10 camadas
          layers.Dense(26, activation="relu"), #56 p/ 10 camadas
          layers.Dense(30, activation="relu"), #63 p/ 10 camadas
          layers.Dense(35, activation="relu"), #70 p/ 10 camadas
          layers.Dense(39, activation="relu"), #78 p/ 10 camadas
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


df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/CICFeatureSelectionPreds50.csv",index=False)


