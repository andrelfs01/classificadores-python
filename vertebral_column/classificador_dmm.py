import pandas as pd
from sklearn.utils import shuffle
import random

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
    #columns = ['sepal_length', 'sepal_width','petal_length','petal_width']
    features = columns
    #print(features)
    if 'class' in features:
        features.remove('class')
    i = 0
    for feature in features:
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
        
def reamostragem(df):
    '''
    (I) Gere um data set balanceado reamostrando as duas classes menores para o tamanho da clase maior. 
    '''
    contagem = df['class'].value_counts()
    classes = df['class'].unique()
    maior_classe = contagem.idxmax()

    for classe in classes:
        if contagem[classe] < contagem[maior_classe]:
            diferenca = (contagem[maior_classe] - contagem[classe])
            print("reamostrar {} elementos na classe {}".format(diferenca,classe))
            population = df[df['class'] == classe]
            resampling = random.choices(population.index.tolist(), k=diferenca)
            for duplicar in resampling:
                sample = population.loc[duplicar].copy()
                #print (sample)
                df = df.append(sample, ignore_index = True)            
    
    contagem = df['class'].value_counts()
    print("nova contagem")
    print (contagem)
    return df

#percorre o dataset e separa treino e teste
#pelvic incidence, pelvic tilt, lumbar lordosis angle, sacral slope, pelvic radius and grade of spondylolisthesis
columns = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle','sacral_slope','pelvic_radius', 'grade_of_spondylolisthesis','class']
#class labels: DH (Disk Hernia), Spondylolisthesis (SL), Normal (NO) and Abnormal (AB)

df_original = pd.read_csv('column_3C.dat', names=columns, header=None, sep=' ',engine='python')
#df_treino = df_original
df_treino = reamostragem(df_original)
#exit()

df_teste = pd.DataFrame(columns=columns)

classes = df_treino['class'].unique()
for i in classes:
    #df_subset = df_treino.iloc[i*50:(i*50)+50].sample(frac=.3)
    df_subset = df_treino[df_treino['class'] == i].sample(frac=.3)
    df_teste = df_teste.append(df_subset)

df_treino = df_treino.drop(df_teste.index)

#EXECUTA A CROSS VALIDACAO
cross_validation(df_treino)

#CALCULA AS  MEDIAS PARA O TESTE DO MODELO
prototipos = df_teste.groupby(df_teste['class']).mean()
#EXECUTA TESTE DO MODELO
test_model(df_teste, prototipos)