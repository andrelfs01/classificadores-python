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

#5-fold
fold1 =  df_treino.sample(frac=.2)
df_treino = df_treino.drop(fold1.index)

fold2 =  df_treino.sample(frac=.2)
df_treino = df_treino.drop(fold2.index)

fold3 =  df_treino.sample(frac=.2)
df_treino = df_treino.drop(fold3.index)

fold4 =  df_treino.sample(frac=.2)
df_treino = df_treino.drop(fold4.index)

fold5 =  df_treino
df_treino = df_treino.drop(fold5.index)

#Vetores prototipos de cada classe
prototipos = df_treino.groupby(df_treino['class']).mean()
print("prototipos")
print(prototipos)




#calcula distancia quadrada para cada prototipos

#escolhe a de menor distancia
result = dmm(prototipos, df_teste)

#gera matriz de confusao

def dmm(prototipos, dataTeste):
    for novo_padrao in dataTeste:
        for class_mean in prototipos:
            distance = mahalanobis_distance(class_mean, novo_padrao)
            
        


