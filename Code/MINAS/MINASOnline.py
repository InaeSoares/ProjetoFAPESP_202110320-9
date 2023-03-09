import pandas as pd
import numpy as np
from dataclasses import dataclass
from math import dist
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#FASE ONLINE

# centro = (1/n) * LS
# Distâncias ao centro:
    # Conjunto Dm={di:di=dist(Xi,centro)}
    # raio = valor máx. Dm [MELHOR] // ou // desv. pad. Dm [PIOR]


# 41 KDD
# 32 KDD FS
# 77 CIC
# 59 CIC FS 

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

@dataclass
class clusterDesconhecido:
    data: list #valor do dado
    cluster: int #cluster ao qual pertence (-1, desconhecido)
    t: float #momento em que foi adicionado ao vetor de desconhecidos
    i: int #posicao no vetor original de dados

#MINAS-ONLINE
def minasOnline(modelo, data, desconhecidos, dado, labelsAtribuidos, tempo, modeloAntigo, ultimaLimpeza):
    numeroDesconhecidos = 50 #50 # quantos valores na memória secundária para detectar novidade
    tempoLimpeza = 200 #100 #30 # quanto tempo entre as limpezas
    clusterDoDado = 0
    labelNovo = bool
    # outdated <- definir -> quantos dados no modelo antigo para esquecer mc
    outdated = 2*len(desconhecidos) # quando fazer limpeza de microclusteres não atualizados do modelo
    # Comparar dado com cada cluster do modelo: 
    # maisProximo <- clusterMaisProximo(EXEMPLO, Modelo) -> devolve o cluster mais próximo do dado
    maisProx, distancia = clusterMaisProximo(data, modelo)
    #SE A DISTANCIA PARA O CLUSTER MAIS PROXIMO FOR MENOR QUE O RAIO DELE, PERTENCE AO CLUSTER
    # Se maisProximo.distancia < maisProximo.cluster.raio:
    if distancia < np.array(maisProx.raio).astype(float):
        # exemplo.rotulo <- maisProximo.cluster.rotulo  
        clusterDoDado = np.array(maisProx.cluster).astype(int) #cluster ao qual o dado pertence
        #informa em qual categoria (0 ou 1) o dado foi classificado  
        #AQUI ADICIONA O LABEL DO DADO COMO O DO CLUSTER AO QUAL ELE PERTENCE
        labelNovo =  maisProx.label 
        # maisProximo.cluster.t <- t.atual      
        labelsAtribuidos[dado] = maisProx.cluster #cluster atribuído é igual ao do grupo do qual o cluster é extensão
        #ATUALIZA O TEMPO DE ADIÇÃO DO ÚLTIMO ELEMENTO AO CLUSTER
        maisProx.t = datetime.timestamp(datetime.now()) #MODELO[MAISPROX.CLUSTER] = DATETIME...
    # Senão:
    else:
        #VERIFICAR SE PERTENCE A ALGUM CLUSTER DO MODELO ANTIGO
        if len(modeloAntigo) > 0:
            maisProx, distancia = clusterMaisProximo(data, modeloAntigo) #estava data[0] e deu erro de iterar sobre 1d array
            #SE SIM, REATIVA CLUSTER
            if distancia < np.array(maisProx.raio).astype(float):
                labelsAtribuidos[dado] = maisProx.cluster #cluster atribuído é igual ao do grupo do qual o cluster é extensão
                #ATUALIZA O TEMPO DE ADIÇÃO DO ÚLTIMO ELEMENTO AO CLUSTER
                maisProx.t = datetime.timestamp(datetime.now())
                reativaCluster(maisProx, modelo, modeloAntigo)
        #SE NÃO, É DESCONHECIDO
        else:
            # exemplo.rotulo <- "desconhecido"
            clusterDoDado = -1 #Pertence a um cluster desconhecido
            # exemplo.tempo = tempo em que foi inserido no vetor de desconhecidos
            # Desconhecidos <- adiciona(EXEMPLO)
            desconhecidos.append(clusterDesconhecido(data, clusterDoDado, datetime.timestamp(datetime.now()), dado))
            #SE TAMANHO DA MEMORIA FOR MAIOR QUE 50, DETECTA NOVIDADE
            # Se (Desconhecidos.tamanho >= NumeroDesconhecidos):
            if len(desconhecidos) >= numeroDesconhecidos:
                #print("detectar novidades")
                # clustersNovidade <- DetectarNovidades (Modelo U ModeloAntigo, Desconhecidos)
                # Modelo <- Modelo U clustersNovidade
                DetectarNovidades(modelo, modeloAntigo, desconhecidos, labelsAtribuidos)       
    #SE PASSOU A JANELA DE TEMPO, LIMPAR MODELOS E MEMÓRIA
    # Se t.atual > (ultimaLimpeza + TempoLimpeza):
    if datetime.timestamp(datetime.now()) > ultimaLimpeza + tempoLimpeza:
        # Desconhecidos <- removeExemplosAntigos (Desconhecidos, ultimaLimpeza)
        removeExemplosAntigos(desconhecidos, tempoLimpeza)
        if len(desconhecidos) > 0 and len(modelo) > outdated:
            # Modelo <- moveModeloAntigo(Modelo, ModeloAntigo, ultimaLimpeza)
            moveModeloAntigo(modelo, modeloAntigo, tempoLimpeza)
        # ultimaLimpeza <- t.atual
        ultimaLimpeza = datetime.timestamp(datetime.now())
    # Saida.adiciona(EXEMPLO)
    return 1

#CLUSTER MAIS PROXIMO - done
def clusterMaisProximo(data, modelo):
    # Exemplo, Modelo
    # proximo <- vazio // o primeiro da lista do modelo
    clusMaisProx = modelo[0]
    centr = np.array(clusMaisProx.centro).astype(float)
    if centr.shape==(1,32):
        centr = centr[0]
    distancia = dist(np.array(data).astype(float), centr)
    #print(clusMaisProx)
    # Para cada mc do Modelo:
    for cluster in range(len(modelo)):
        centro = np.array(modelo[cluster].centro).astype(float)
        #print('CENTRO:', centro)
        # Se proximo > distanciaEuclidiana(exemplo, mc)
        mp = np.array(clusMaisProx.centro).astype(float)
        if mp.shape == (1,32):
            mp = mp[0]
        if centro.shape == (1,32):
            centro = centro[0]
        if abs(dist(np.array(data).astype(float), mp)) > abs(dist(np.array(data).astype(float), np.array(centro).astype(float))):
            # proximo <- distanciaEuclidiana(exemplo, mc)
            clusMaisProx = modelo[cluster]
            distancia = dist(np.array(data).astype(float), np.array(centro).astype(float))
        # Senao:
            #continue
    # proximo.cluster
    return clusMaisProx, distancia

#DETECTAR NOVIDADES - done
def DetectarNovidades(modelo, modeloAntigo, desconhecidos, labelsAtribuidos):
    # minExemplos <- de acordo com artigo -> min de exemplos apra considerar como mc
    minExemplos = 30
    # fatorNovidade <- de acordo com artigo 
    fatorNovidade = 1.1
    # novoModelo <- vazio
    novoModelo = [] # modelo caso exista nova classe e seja feita atualização 
    novoModeloData = [] # modelo caso exista nova classe e seja feita atualização - só o vetor de dado
    #pegar apenas os valores numéricos dos dados desconhecidos
    valoresDesconhecidos = [] #vetor para receber os valores para clusterização
    silhuetas = [] #vetor para armazenar silhuetas e definir melhor número de clusteres
    aux = microCluster
    #PEGAR SÓ OS VALORES DO VETOR DE DESCONHECIDOS, SEM AS OUTRAS INFORMAÇÕES
    for desconhecido in desconhecidos:
        valoresDesconhecidos.append(np.array(desconhecido.data).astype(float))
    # clusterizacao(Desconhecidos)
    #TESTAS A CLUSTERIZAÇÃO PARA TODOS NÚMEROS POSSÍVEIS DE CLUSTERES DENTRO DE DESCONHECIDOS
    for i in range(len(desconhecidos)):
        if i == 0 or i == 1:
            continue
        else:
            # -- GERAR LABLE DO CLUSTER PARA CADA dado do vetor
            clusterLabels = KMeans(n_clusters = i).fit_predict(valoresDesconhecidos)
            # -- CALCULAR SILHUETA DE CADA CLUSTER POSSÍVEL E COLOCAR NUM VETOR
            silhuetas.append(silhouette_score(valoresDesconhecidos, clusterLabels))
    # -- PEGAR MAIOR VALOR DE SILHUETA -> NUMERO DE CLUSTERES FORMADOS
    maxSilhueta = max(silhuetas)
    #print(maxSilhueta)
    # -- APLICAR KMEANS COM ESTE VALOR
    clusterLabels = KMeans(n_clusters = silhuetas.index(maxSilhueta)+2).fit_predict(valoresDesconhecidos)
    #print(clusterLabels)
    # -- SEGUIR COM  O ALGORITMO
    # Para cada novo cluster gerado na clusterização:
    nElementos = [] #quantos elementos tem de cada cluster
    #CRIAR UM VETOR COM NÚMERO DE ELEMENTOS DE CADA CLUSTER E O RESPECTIVO CLUSTER
    for cluster in range(len(clusterLabels)):
        nElementos.append((clusterLabels.tolist().count(clusterLabels[cluster]), clusterLabels[cluster]))
    nElementos = [*set(nElementos)] #remove duplicatas
    #print(nElementos)
    #print(nElementos[0][1])
    clusteresCriados = [] #para limpar depois das variáveis da memória
    #PARA CADA ELEMENTO DENTRE OS PARES (NUMERO, CLUSTER)
    for clusterNovo in nElementos:
        clusVelho=False
        #SEPARAR TODOS OS VALORES QUE PERTENCEM AO DETERMINADO CLUSTER
        for dado in range(len(desconhecidos)): #para cada dado no vetor de desconhecidos
            if clusterLabels[dado]==clusterNovo[1]: #se o correspondente na lista de clusteres for igual ao cluster analisado
             #PEGAR EM desconhecidos OS VALORES QUE TEM O NUMERO DO CLUSTER EM clusterLabels
             novoModelo.append(desconhecidos[dado]) #adiciona o dado ao vetor de dados com o mesmo label
             novoModeloData.append(np.array(desconhecidos[dado].data).astype(float))
        aux  = criaNovidade(novoModelo, modelo)
        #se novo.tamanho >= minExemplos ^ novo.silhueta > 0: 
        # SE O CLUSTER NOVO TEM NUMERO NECESSÁRIO DE EXEMPLOS E A SILHUETA DO CLUSTER É MAIOR QUE ZERO
        if clusterNovo[0] >= minExemplos and silhueta(aux, novoModeloData, modelo) > 0: #PEGAR SILHUETA DO SAMPLE!!!
            clusteresCriados.append(clusterNovo[1])
            # maisProximo <- clusterMaisProximo(novo, Modelo)
            maisProx, dist = clusterMaisProximo(np.array(aux.centro[0]).astype(float), modelo) 
            # se maisProximo.distancia < (maisProximo.cluster.raio x fatorNovidade):
            if dist <= np.array(maisProx.raio).astype(float)*fatorNovidade:
                # novo.rotulo <- maisProximo.cluster.rotulo
                #AQUI ADICIONAR LABEL DO DADO COMO O DO CLUSTER
                for a in novoModelo:
                    labelsAtribuidos[a.i] = maisProx.cluster #cluster atribuído é igual ao do grupo do qual o cluster é extensão
                # novo.tipo <- extensao
            # senao
            elif len(modeloAntigo) > 0:
                maisProx1, dist1 = clusterMaisProximo(np.array(aux.centro[0]).astype(float), modeloAntigo)    
                # se maisProximo.distancia < (maisProximo.cluster.raio x fatorNovidade):
                if dist1 < np.array(maisProx1.raio).astype(float)*fatorNovidade:
                    clusVelho=True
                    #REATIVAR MODELO ANTIGO
                    reativaCluster(maisProx1, modelo, modeloAntigo)
                    #DAR LABEL PARA O CLUSTER
                    for a in novoModelo:
                        labelsAtribuidos[a.i] = maisProx1.cluster
    
            if not clusVelho and (dist > np.array(maisProx.raio).astype(float)*fatorNovidade):
                for a in novoModelo:
                    labelsAtribuidos[a.i] = len(modelo) #cluster atribuído é igual ao do grupo do qual o cluster é extensão
                # novo.rotulo <- proximaNovidade
                # proximaNovidade < proximaNovidade + 1
                # novo.tipo <- "novidade"
                # novoModelo <- novoModelo U novo
                #adicionaNovidade(novoModelo, modelo) 
                modelo.append(aux)
    
    # REMOVER EXEMPLOS DO CLUSTER DA MEMÓRIA 
    # Desconhecidos <- Desconhecidos - novo.exemplos
    # remover de DESCONHECIDOS os valores referentes ao cluster nElementos[1] em CLUSTERLABELS
    # PEGAR VALOR CLUSTERNOVO[1], REMOVER ESSE VALOR DO CLUSTERLABELS E DO DESCONHECIDOS
    #for criado in clusteresCriados:
    #    a = clusteresCriados.index(criado)
    #    desconhecidos.remove(np.array(desconhecidos[a]))
    #    clusterLabels.tolist().remove(clusterLabels[a])

    for b in range(len(clusterLabels[:])):
        for c in range(len(clusteresCriados[:])):
            if clusterLabels[b] == np.array(clusteresCriados[c]): # erro la de a.all() e a.any()
                desconhecidos[b] = -1
                clusterLabels[b] = -1
    desconhecidos[:] = (value for value in desconhecidos if value != -1)
    (value for value in clusterLabels.tolist() if int(value) != -1)
            
    return 1

#ADICIONA O CLUSTER NOVIDADE AO MODELO - done
def criaNovidade(novoModelo, modelo):
    #calcular tamanho do modelo (tamanho-1 = numero do último cluster)
    clustNovidade = len(modelo)
    novoCluster = microCluster(0, np.zeros((1, 32), float), np.zeros((1, 32), float), np.zeros((1, 32), float), 0.0, 0.0, True, 'C') #exemplo vazio para criar novo cluster 
    #criar variável do cluster com os dados do novo cluster
    for novo in novoModelo:
        novoCluster.n = novoCluster.n + 1
        novoCluster.ls = np.array(novoCluster.ls).astype(float) + np.array(novo.data).astype(float)
        novoCluster.ss = np.array(novoCluster.ls).astype(float) + np.array(novo.data).astype(float) #ELEVAR AO QUADRADO!!!
    novoCluster.centro = (1/novoCluster.n) * novoCluster.ls 
    for novo in novoModelo: 
        if novoCluster.raio < abs(dist(novo.data.astype(float), np.array(novoCluster.centro[0]).astype(float))): 
                novoCluster.raio = abs(dist(novo.data.astype(float), np.array(novoCluster.centro[0]).astype(float)))
    novoCluster.t = datetime.timestamp(datetime.now())
    novoCluster.cluster = clustNovidade
    #adicionar ao modelo
    #modelo.append(novoCluster)

    return novoCluster

#SILHUETA -> USAR CÁLCULO DA SCIKIT - done
def silhueta(clusterNovo, novoModeloData, modelo):
    valorSilhueta = 0.0
    distanciasElementosCentro = []
    for elemento in novoModeloData:
        distanciasElementosCentro.append(abs(dist(elemento, np.array(clusterNovo.centro[0]).astype(float))))
    # a = SD{dist(Xi, centro)} -> standard deviation das distancias dos elementos do novo mc e o centroide dele
    a = np.std(distanciasElementosCentro)

    # b = min{dist(centroNovo, centro(M)): M E Modelo} -> distancia entre o centro do novo mc e do mc mais proximo
    _, b = clusterMaisProximo(np.array(clusterNovo.centro[0]).astype(float), modelo)

    # Silhueta = (b-a)/max(a,b)
    valorSilhueta = (b-a)/max(a,b)
    return valorSilhueta

#REMOVE EXEMPLOS ANTIGOS - done
def removeExemplosAntigos(desconhecidos, tempoLimpeza):
    # Desconhecidos, tempoLimpeza
    # t.atual - Desconhecidos.exemplo.t >= tempoLimpeza :
            # Desconhecidos - exemplo 
    i = 0
    for desconhecido in desconhecidos:
        if datetime.timestamp(datetime.now()) - desconhecido.t >= tempoLimpeza:
            del desconhecidos[i]
        i += 1
    return 1

#MOVE MODELO ANTIGO - done
#Remove clusters que não são atualizados há muito tempo, considerados obsoletos
def moveModeloAntigo(modelo, modeloAntigo, tempoLimpeza):
    #SE ELEMENTO DO MODELO ATUAL.T NÃO É ATUALIZADO HA MAIS TEMPO Q A JANELA DE LIMPEZA
    for cluster in modelo:
        if cluster.t > 0.0 and datetime.timestamp(datetime.now()) - cluster.t > tempoLimpeza:
            # ADICIONA ELEMENTO NO MODELO ANTIGO
            modeloAntigo.append(cluster)
            #REMOVE ELEMENTO DO MODELO ATUAL
            modelo.remove(cluster)
    return 1
 
#REATIVA CLUSTERES ANTIGOS DO MODELO ANTIGO - done
def reativaCluster(clustReativar, modelo, modeloAntigo):
    modelo.append(clustReativar)
    modeloAntigo.remove(clustReativar)
    return 1

def main():
    # Inputs: Modelo, stream de dados
    # MODELO KDD
    #dataframeModelo = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOff.csv", header=None)
    # MODELO CSECIC
    #dataframeModelo = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOffcic.csv", header=None)
    #MODELO CSECIC FS
    #dataframeModelo = pd.read_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOffcicFS.csv")
    #MODELO KDD99 FS
    dataframeModelo = pd.read_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/modeloMinasOffkddFS.csv")
    raw_data_model = dataframeModelo.values
    dataframeModelo.head()
    raw_data_model = raw_data_model[1: , :]
    #print(raw_data)
    modelo=[]
    for i in range(len(raw_data_model)):
        aux=microCluster(0, np.zeros((1, 32), float), np.zeros((1, 32), float), np.zeros((1, 32), float), 0.0, 0.0, True, 'C')
        aux.n = raw_data_model[i][0]
        aux.ls = raw_data_model[i][1].translate({ord(i): None for i in '\n[]\''})
        aux.ls = aux.ls.replace(' ', ',')
        aux.ls = aux.ls.split(',')
        aux.ss = raw_data_model[i][2].translate({ord(i): None for i in '\n[]\''})
        aux.ss = aux.ss.replace(' ', ',')
        aux.ss = aux.ss.split(',')
        aux.centro = raw_data_model[i][3].translate({ord(i): None for i in '\n[]\''})
        aux.centro = aux.centro.replace(' ', ',')
        aux.centro = aux.centro.split(',')
        aux.raio = raw_data_model[i][4]
        aux.t = raw_data_model[i][5] #.astype(float)
        aux.label = raw_data_model[i][6] #.astype(bool) 
        aux.cluster = raw_data_model[i][7]
        modelo.append(aux)

    # CARREGAR DADOS KDD
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/data400kddDICnormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/data800kddDICnormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/data1300kddDICnormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/data5000kddDICnormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/FeatureSelection/400kdd75chi2.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/FeatureSelection/800kdd75chi2.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/FeatureSelection/1300kdd75chi2.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/KDD99/Binary/FeatureSelection/5000kdd75chi2.csv", header=None) V

    # CARREGAR DADOS CSECIC
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/400cicNormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/800cicNormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/1300cicNormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/DAWOUD/Autoencoder/5000cicNormalized.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/400cic75chi2.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/800cic75chi2.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/1300cic75chi2.csv", header=None) V
    #dataTeste = pd.read_csv("C:/Users/Administrator/Downloads/ICInae/Datasets/CICAWS/FeatureSelection/5000cic75chi2.csv", header=None) V

    dataTeste = dataTeste.values

    # SELECIONAR LABLES, NA ÚLTIMA COLUNA
    labelsTeste = dataTeste[1:, -1]
    # COLUNAS DOS DADOS A SEREM USADAS PARA O TREINAMENTO
    dataTeste = dataTeste[1:, :32]
    #dataTeste = dataTeste[:,:-1] #para 5000 dados
    #print(dataTeste[0])
    #print(dataTeste[1])

    labelsAtribuidos = [] #armazena os labels dos clusteres atribuídos a cada dado
    for _ in range(len(dataTeste)): #inicia um vetor do tamanho do conjunto de dados para que possa ser acessado por posição
        labelsAtribuidos.append(-1) #valor menor que zero para fazer verificação se todos foram classificados no final
    
    #VARIÁVEIS 'GLOBAIS'
    # Desconhecidos <- vazio 
    desconhecidos = [] #short term memory
    # ModeloAntigo <- vazio
    modeloAntigo = []
    # ultimaLimpeza <- 0
    ultimaLimpeza = datetime.timestamp(datetime.now())
    tempo = datetime.timestamp(datetime.now())

    for dado in range(len(dataTeste)):
        #print(dataTeste[dado])
        minasOnline(modelo, dataTeste[dado].astype(float), desconhecidos, dado, labelsAtribuidos, tempo, modeloAntigo, ultimaLimpeza)
        #print(dataTeste[dado])
        #print(desconhecidos)
        print(dado)

    #VERIFICAR SE ALGUM VALOR NÃO FOI CLASSIFICADO (SE FOI DESCARTADO DA MEMÓRIA, POR EXEMPLO)
    #VALORES AGRUPADOS EM UM CLUSTER FINAL, PARA RECEBEREM UM VALOR PARA CLASSIFICAÇÃO
    descartados = len(modelo)
    for classificados in range(len(labelsAtribuidos)):
        if labelsAtribuidos[classificados] == -1:
            labelsAtribuidos[classificados] = descartados

    #SALVAR VALORES
    df = pd.DataFrame(dataTeste)
    #JUNTAR VETOR DE CLUSTERES ATRIBUIDOS NO FINAL DO CONJUNTO DE DADOS
    df["Cluster"] = labelsAtribuidos
    #ADICIONAR LABELS DE VOLTA
    df["Label"] = labelsTeste

    # SALVAR KDD
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd5000.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd5000FS.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd1300.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd1300FS.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd800.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd800FS.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd400.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINASkdd400FS.csv",index=False) V

    # SALVAR CIC
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic5000.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic5000FS.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic1300.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic1300FS.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic800.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic800FS.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic400.csv",index=False) V
    #df.to_csv(r"C:/Users/Administrator/Downloads/ICInae/DAWOUD/MINAS/MINAScic400FS.csv",index=False) V
    print('saved')


main()