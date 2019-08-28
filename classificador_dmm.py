import pandas as pd
#DMM

#percorre o dataset e separa treino e teste
columns = ['sepal_length', 'sepal_width','petal_length','petal_width','class']
df_treino = pd.read_csv('iris.data', names=columns)

df_teste = pd.DataFrame(columns=columns)
for i in range(3):
    #print("retirando teste...")
    df_subset = df_treino.iloc[i*50:(i*50)+50].sample(frac=.3)
    #print(df_subset)
    df_teste = df_teste.append(df_subset)

df_treino = df_treino.drop(df_teste.index)

folds = kfold(df_treino, 5)

#Vetores prototipos de cada classe
prototipos = df_treino.groupby(df_treino['class']).mean()
print("prototipos")
print(prototipos)

#calcula distancia quadrada para cada prototipos
n
#escolhe a de menor distancia
result = dmm(prototipos, df_teste)

#gera matriz de confusao

def dmm(prototipos, dataTeste):
    for novo_padrao in dataTeste:
        for class_mean in prototipos:
            distance = mahalanobis_distance(class_mean, novo_padrao)
            
import random
def kfold(dataset, k, seed = 42):
    
    size = len(dataset)
    subset_size = round(size / k)
    random.Random(seed).shuffle(dataset)
    subsets = [dataset[x:x+subset_size] for x in range(0, len(dataset), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train,test))
        
    return kfolds