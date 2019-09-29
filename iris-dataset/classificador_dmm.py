import pandas as pd
from sklearn.utils import shuffle

#DMM CLASSIFICADOR USANDO A MENOR DISTANCIA DO PADRAO ATE OS PONTOS MÉDIOS DAS POSSIVEIS CLASSES 
def dmm(prototipos, dataTeste):
    for i in dataTeste.index:
        distance = -1
        for c in prototipos.index:
            r = distancia_quadrada(prototipos.loc[c],dataTeste.loc[i])
            if (distance == -1 or r < distance):
                dataTeste.loc[i, 'classified'] = c
                distance = distancia_quadrada(prototipos.loc[c], dataTeste.loc[i])
    return dataTeste

#CALCULO DE DISTANCIA QUANDRADA ENTRE DOIS PADROES
def distancia_quadrada(a, b):
    columns = ['sepal_length', 'sepal_width','petal_length','petal_width']
    i = 0
    for feature in columns:
        i = i + ((a[feature] - b[feature]) ** 2 )
    return i

#DIVISAO DE DADOS EM K FOLDS  - RETORNA UMA LISTA DE K PARES <TREINO, TESTE> SEGUINDO A IDEIA DA CROSS VALIDACAO 
def kfold(dataset, k, seed):
    size = len(dataset)
    subset_size = round(size / k)
    dataset = shuffle(dataset)
    subsets = []
    for x in range(0,k):
        subsets.append(dataset.iloc[x*subset_size:(x*subset_size)+subset_size])
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = pd.DataFrame(columns=columns)
        for subset in subsets:
            if not test.equals(subset):
                train = train.append(subset)
        kfolds.append((train,test))
        
    return kfolds

#gera matriz de confusao
def get_confusion_matrix(data, expected, classified):
    classes = data[expected].unique()
    classes.sort()
   
    result = pd.DataFrame(columns=classes)

    for expected_class in classes:
        for classified_class in classes:
            i = data[(data[expected] == expected_class) & (data[classified] == classified_class)]
            result.loc[expected_class, classified_class] =  i.shape[0]

    return result

#EXECUTA VALIDACAO - CROSS VALIDACAO EM 5 FOLDS E MOSTRA AS MATRIZES E UMA MATRIZ DE MÉDIA 
def cross_validation(df_treino):
    folds = kfold(df_treino, 5, 1 )
    all_matrix = []
    cont = 0
    for f in folds:
        treino, validacao = f
        #Vetores prototipos de cada classe
        prototipos = treino.groupby(treino['class']).mean()
        result = dmm(prototipos, validacao)
        matrix = get_confusion_matrix(result, 'class','classified')
        all_matrix.append(matrix)
        print("Fold {}:".format(cont))
        print(matrix)
        print("")
        cont+=1

    mean_matrix = pd.concat(all_matrix).groupby(level=0).mean()
    print("Mean cross validation:")
    print(mean_matrix)
    print("")

#TESTE DE MODELO APLICANDO O DMM
def test_model(df_teste, prototypes):
    result = dmm(prototypes, df_teste)
    matrix = get_confusion_matrix(result, 'class','classified')
    print("Test model result:")
    print(matrix)
    print("")
        

#percorre o dataset e separa treino e teste
columns = ['sepal_length', 'sepal_width','petal_length','petal_width','class']
df_treino = pd.read_csv('iris.data', names=columns)

df_teste = pd.DataFrame(columns=columns)
for i in range(3):
    df_subset = df_treino.iloc[i*50:(i*50)+50].sample(frac=.3)
    df_teste = df_teste.append(df_subset)

df_treino = df_treino.drop(df_teste.index)

#EXECUTA A CROSS VALIDACAO
cross_validation(df_treino)

#CALCULA AS  MEDIAS PARA O TESTE DO MODELO
prototipos = df_teste.groupby(df_teste['class']).mean()
#EXECUTA TESTE DO MODELO
test_model(df_teste, prototipos)