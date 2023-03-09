import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass
from math import dist
from collections import Counter
import statistics
from datetime import datetime

#FASE DE TREINAMENTO 

# Clusterização
# K-means ou
# CluStream

#Input: Conjunto de treinamento rotulado
    # Conjunto é dividido e grupos de acordo com os labels (2, para 0 e para 1)
        # Para cada grupo:
            #Aplicar clusterização
            #Output: Micro-clusters (com tag do grupo inicial ao qual pertence), centro de cada cluster
                # Micro-cluster: (n, LS, SS, t)
                    # n > quantos exemplos tem no mc
                    # LS > vetor soma linear dos exemplos do mc
                    # SS > vetor soma dos quadrados dos exemplos do mc
                    # t > instante de chegada do último exemplo classificado (0, até a fase offline)
                # Modelo de decisão é a junçaõ de todos os micro-clusters


#############
# FASE OFFLINE
#############

# 41 KDD
# 32 KDD FS
# 77 CIC
# 59 CIC FS   

# CARREGAR DADOS PARA TREINAMENTO
#dataframe = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/kdd10k.csv", header=None)
#dataframe = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/cic10kNormal.csv", header=None)
#dataframe = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/10kcic75chi2.csv", header=None)
dataframe = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/FeatureSelection/10kkdd75chi2.csv", header=None)
raw_data = dataframe.values
dataframe.head()

# SELECIONAR LABLES, NA ÚLTIMA COLUNA
labels = raw_data[1:, -1]

# COLUNAS DOS DADOS A SEREM USADAS PARA O TREINAMENTO
data = raw_data[1:, 0:32] 

# SEPRAR DADOS NORMAIS DOS DADOS ANÔMALOS
labels = labels.astype(float)
labels = labels.astype(bool)
normal_data = data[~labels] #dados (false) normais (0)
anomalous_data = data[labels] #dados (true) anomalos (1)
lableTrue=[]
lableFalse=[]

#dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/data400kddDICnormalized.csv", header=None)
#dataTeste = dataTeste.values
##dataTeste.head()
## SELECIONAR LABLES, NA ÚLTIMA COLUNA
#labelsTeste = dataTeste[1:, -1]
## COLUNAS DOS DADOS A SEREM USADAS PARA O TREINAMENTO
#dataTeste = dataTeste[1:, 0:41]

print('carregou todos os dados')

for i in range(len(labels)):
    if (labels[i]==0):
        lableTrue=np.append(lableTrue, labels[i])
    elif (labels[i]==1).all():
        lableFalse=np.append(lableFalse, labels[i])

print(lableFalse)
print(lableTrue)

#print(normal_data.shape)
#print(anomalous_data.shape)

# DEFINIR O MODELO
@dataclass
class microCluster:
    n: int #quantidade de elementos no microcluster
    ls: list #somas linear dos elementos 
    ss: list #soma quadrada dos elementos
    centro: list #vetor centro do cluster
    raio: float #raio do microcluster
    t: float #tempo da última adição de elementos ao microcluster
    label: bool #label atribuído ao cluster
    cluster: int #número do cluster

# FUNÇÃO PARA SEPARAR MICRO-CLUSTERS
def separateMCs(dataset):
    kmeans=KMeans(n_clusters=100, random_state=0).fit(dataset)
    indexes=kmeans.labels_
    #print(dataset)
    #print(indexes)
    indexes = np.transpose(indexes)
    print(indexes.shape)
    print('separated\n')
    return kmeans, indexes

# FUNÇÃO PARA DEFINIR OS PARÂMETROS DE CADA MICRO-CLUSTER  
# MODELO = [n, LS, SS, t, label]
def createMCs(indexes, dataset, labels, model):
    mc = microCluster(0, np.zeros((1, 32), float), np.zeros((1, 32), float), np.zeros((1, 32), float), 0.0, 0.0, True, 0)
    for j in (range(len(indexes))):
        mc.n = 1
        mc.ls = dataset[j].astype(float)
        mc.ss = pow(dataset[j].astype(float), 2)
        mc.t = 0.0
        mc.centro = (1/mc.n) * mc.ls
        mc.raio = 0.0
        mc.label = labels[j]
        mc.cluster = 0
        if (model[indexes[j]].n==0):
            model[indexes[j]]=mc
        else:
            aux = microCluster(model[indexes[j]].n, model[indexes[j]].ls, model[indexes[j]].ss, model[indexes[j]].centro, model[indexes[j]].raio, model[indexes[j]].t, model[indexes[j]].label, model[indexes[j]].cluster)
            aux.n += 1
            aux.ls = model[indexes[j]].ls + mc.ls
            aux.ss = model[indexes[j]].ss + mc.ss
            aux.centro = (1/aux.n) * aux.ls
            if aux.raio < abs(dist(dataset[j].astype(float), aux.centro)):
                aux.raio = abs(dist(dataset[j].astype(float), aux.centro))
            aux.label = labels[j]
            aux.cluster = 0
            model[indexes[j]] = aux
    print('model\n')
    return model
    # juntar a lista de clusteres com a tabela de exemplos
        # criar os vetores de cada cluster
            # somar os valores dos clusters iguais e adicionar na tabela do modelo
            #Output: Micro-clusters (com tag do grupo inicial ao qual pertence), centro de cada cluster
                # Micro-cluster: (n, LS, SS, t)
                    # n > quantos exemplos tem no mc
                    # LS > vetor soma linear dos exemplos do mc
                    # SS > vetor soma dos quadrados dos exemplos do mc
                    # t > instante de chegada do último exemplo classificado (0, até a fase offline)


# FUNÇÃO PARA SALVAR O MODELO
def saveModel(modeloV): #, modeloF
    modelo = modeloV #np.append(modeloV, modeloF)
    dataframe=[ [[] for col in range(8)] for col in range(len(modelo))]
    for k in range(len(modelo)):
        #print(modelo[i].n)
        dataframe[k][0]=modelo[k].n
        dataframe[k][1]=modelo[k].ls.astype(str)
        dataframe[k][2]=modelo[k].ss.astype(str)
        dataframe[k][3]=modelo[k].centro.astype(str)
        dataframe[k][4]=modelo[k].raio
        dataframe[k][5]=modelo[k].t
        dataframe[k][6]=modelo[k].label
        dataframe[k][7]=k
    df = pd.DataFrame(dataframe) 
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOff.csv",index=False)
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOffcic.csv",index=False)
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOffcicFS.csv",index=False)
    df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOffkddFS.csv",index=False)
    print('saved')
    # Modelo de decisão é a junçaõ de todos os micro-clusters

exmc=microCluster(0, np.zeros((1, 32), float), np.zeros((1, 32), float), np.zeros((1, 32), float), 0.0, 0.0, True, 0)
modeloNormal=[exmc]*100
#print(modeloNormal)
#modeloAnormal=[exmc]*300
print('modelo criados\n')
kMeans, indexes = separateMCs(data)
#print(normalIndexes)
modeloNormal = createMCs(indexes, data, labels, modeloNormal)
print('normalCreated\n')

#_, anomalousIndexes = separateMCs(anomalous_data)
#print(anomalousIndexes)
#modeloAnormal = createMCs(anomalousIndexes, anomalous_data, lableFalse, modeloAnormal)
#print('anomalousCreated\n')

saveModel(modeloNormal) #, modeloAnormal

