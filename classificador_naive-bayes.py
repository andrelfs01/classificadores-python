import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from collections import defaultdict
from math import e
from math import pi

#naive_bayes CLASSIFICADOR POR GAUSSIANA - CALCULO DE PROBABILDIADES A POSTERIORI E SELECIONA A MAIOR
def naive_bayes(sumario_classes, dataTeste):
    for i in dataTeste.index:
        aposteriori_probs = posteriori(sumario_classes, dataTeste.loc[i])
        # print(posterior_probs)
        # print(type(posterior_probs))
        maior_prob = max(aposteriori_probs, key=aposteriori_probs.get)
        dataTeste.loc[i, 'classified'] = maior_prob
    return dataTeste

#PROBABILIDADE DE X APLICADA NA GAUSSIANA (NORMAL)
def gaussiana(x, media, desvio_padrao):
    variancia = desvio_padrao ** 2
    diferenca_quadrada = (x - media) ** 2
    expoente = -diferenca_quadrada / (2 * variancia)
    numerador = e ** expoente
    denominador = ((2 * pi) ** .5) * desvio_padrao
    normal_prob = numerador / denominador
    return normal_prob

#A PRIORI DE CADA CLASSE
def priori(data, col):
    return data[col].value_counts(normalize=True)

#USA GAUSSINA PARA CADA FEATURE E MULTIPLICA PELA PROB A PRIORI | verossimilhança?
def calculo_probabilidade_conjunta(sumario_classes, pattern):
    joint_probs = {}
    for target, features in sumario_classes.items():
        total_features = len(features['summary'])
        verossimilhanca = 1
        for index in range(total_features):
            feature = pattern[index]
            media = features['summary'][index]['media']
            desvio = features['summary'][index]['desvio']
            normal_prob = gaussiana(feature, media, desvio)
            verossimilhanca *= normal_prob
        apriori_prob = features['apriori_prob']
        joint_probs[target] = apriori_prob * verossimilhanca
    return joint_probs

# P(c1 | x) PARA TODAS A CLASSES => P(novo_dado | classe_i) para cada classe i
def posteriori(sumario_classes, pattern):
        posterior_probs = {}
        prob_conjunta = calculo_probabilidade_conjunta(sumario_classes, pattern)
        # print(prob_conjunta)
        # print(type(prob_conjunta))
        marginal_prob = conjunta_total(prob_conjunta)
        for target, joint_prob in prob_conjunta.items():
            posterior_probs[target] = joint_prob / marginal_prob
        return posterior_probs

#calcula media de cada feature
def media(lista):
        result = sum(lista) / float(len(lista))
        return result

#calcula devio padrao de cada feature para cada classe
def desvio_padrao(lista):
    #print("lista desvio: {}".format(lista))
    media_lista = media(lista)
    #print("media desvio: {}".format(lista))
    lista_diferenca_quadrada = []
    for num in lista:
        squared_diff = (num - media_lista) ** 2
        lista_diferenca_quadrada.append(squared_diff)
    squared_diff_sum = sum(lista_diferenca_quadrada)
    sample_n = float(len(lista) - 1)
    var = squared_diff_sum / sample_n
    return var ** .5

#AGRUPA OS DADOS DE TREINO POR CLASSE E REMOVE A COLUNA class
def group_by_class(dataTraino, expected):
    target_map = defaultdict(list)
    for index in dataTraino.index:
        features = dataTraino.loc[index]
        x = features[expected]
        target_map[x].append(features.iloc[:-1])
    return dict(target_map)

#SOMA DE TODAS CONJUNTAS
def conjunta_total( probs_conjuntas):
        """
        conjunta_total =
          [P(setosa) * P(sepal length | setosa) * P(sepal width | setosa) * P(petal length | setosa) * P(petal width | setosa)]
        + [P(versicolour) * P(sepal length | versicolour) * P(sepal width | versicolour) * P(petal length | versicolour) * P(petal width | versicolour)]
        + [P(virginica) * P(sepal length | verginica) * P(sepal width | verginica) * P(petal length | verginica) * P(petal width | verginica)]
        """
        conjunta = sum(probs_conjuntas.values())
        return conjunta

#CALCULA MEDIA E DESVIO PADRAO DE CADA FEATURE DE CADA CLASSE USANDO O CONJUNTO DE TREINO
def summarizar(dataTreino):
    for feature in zip(*dataTreino):
        yield {
            'desvio': desvio_padrao(feature),
            'media': media(feature)
        }

#AGRUPA OS DADOS E SUMARIZA OS DADOS POR CLASSE
# SUMARIO: [{'CLASSE':A,  'FEATURES': [{'MEDIA': X , 'DESVIO': Y} ....], 'APRIORI': p}, ...]
def treinar(dataTreino, target):
        group = group_by_class(dataTreino, target)
        #print("group {}".format(group))
        summaries = {}
        for target, features in group.items():
            apriori_probs = priori(dataTreino, 'class')
            #print("priori {}".format(priori_probs))
            summaries[target] = {                
                'apriori_prob': apriori_probs.loc[target],
                'summary': [i for i in summarizar(features)],
            }
        return summaries


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
        sumarizado = treinar(treino, 'class')
        #print("result sumarizacao: {}".format(sumarizado))
        result = naive_bayes(sumarizado, validacao)
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

#TESTE DE MODELO APLICANDO O naive_bayes
def test_model(df_treino, df_teste):
    sumarizado = treinar(df_treino, 'class')
    result = naive_bayes(sumarizado, df_teste)
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

#EXECUTA TESTE DO MODELO
test_model(df_treino, df_teste)