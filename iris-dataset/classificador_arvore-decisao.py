import pandas as pd
from sklearn.utils import shuffle
import numpy as np


# Realiza a divisao de um corte dado, feature e valor
def testa_corte(feature, valor, dataset):
	left, right = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
	for row in dataset.iterrows():
		# print("teste {} < {}".format(row[1][feature], valor))
		if row[1][feature] < valor:
			left.loc[row[0]] = row[1]
		else:
			# right.append(row)
			right.loc[row[0]] = row[1]
	return (left, right)

# Calculo do indice de Gini


def gini_index(grupos, classes):
	# quantidade de elementos em cada grupo do corte
	n_instances = float(sum([len(group) for group in grupos]))
	# calculo do indice de gini
	gini = 0.0
	for grupo in grupos:
		size = float(len(grupo))
		# evita erro do grupo com 0
		if size == 0:
			continue
		score = 0.0
		# indice do grupo, analisando o indice para cada classe
		for class_val in classes:
			saida = grupo['class'].value_counts()
			if hasattr(saida, class_val):
				p = saida.loc[class_val] / size
			else:
				p = 0
			score += p * p
		# ponderando o indice pelo tamenho do grupo
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Selecina corte ideal com gini_index


def seleciona_corte(dataset):

	class_values = list(dataset['class'].unique())
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	indexes = list(dataset.columns)
	indexes.remove('class')

	for index in range(len(indexes)):
		for row in dataset.iterrows():
			groups = testa_corte(index, row[1][index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = indexes[index], row[1][index], gini, groups
	return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Verifica no folha


def to_terminal(group):
	# print("final grupo {}".format(group))
	saida = group['class'].value_counts()
	# print("saida")
	# print(saida)
	# print(saida.idxmax())
	maior_contagem = saida.idxmax()
	return maior_contagem


# Decisão de corte ou no folha
def corte(no, max_depth, min_size, depth):
	left, right = no['groups']
	del(no['groups'])
	# checa se algum grupo esta vazio
	if left.empty or right.empty:
		no['left'] = no['right'] = to_terminal(left.append(right))
		return
	# checa se atingiu a profundidade maxima
	if depth >= max_depth:
		no['left'], no['right'] = to_terminal(left), to_terminal(right)
		return
	# se o grupo esquedo é menor que o tamanho minimo para de cortar
	if len(left) <= min_size:
		no['left'] = to_terminal(left)
	# senao corta para diminuir
	else:
		no['left'] = seleciona_corte(left)
		corte(no['left'], max_depth, min_size, depth+1)
	# se o grupo direito é menor que o tamanho minimo para de cortar
	if len(right) <= min_size:
		no['right'] = to_terminal(right)
	# senao corta para diminuir
	else:
		no['right'] = seleciona_corte(right)
		corte(no['right'], max_depth, min_size, depth+1)


# gera um nó da arvore
def gera_arvore(treino, max_depth, min_size):
	raiz = seleciona_corte(treino)
	corte(raiz, max_depth, min_size, 1)
	return raiz

# Aplica um padrao na arvore
def classificacao(arvore, padrao):
	if padrao[arvore['index']] < arvore['value']:
		if isinstance(arvore['left'], dict):
			return classificacao(arvore['left'], padrao)
		else:
			return arvore['left']
	else:
		if isinstance(arvore['right'], dict):
			return classificacao(arvore['right'], padrao)
		else:
			return arvore['right']

# Contrutor da arvore de decisao


def decision_tree(treino, teste, max_depth, min_size):
	arvore = gera_arvore(treino, max_depth, min_size)
	print("arvore {}".format(arvore))
	print("")
	print("")
	for row in teste.iterrows():
		prediction = classificacao(arvore, row[1])
		teste.loc[row[0], 'classified'] = prediction
	#print("final: {}".format(teste))
	return(teste, arvore)


def decision_tree_classification(arvore, teste):
	for row in teste.iterrows():
		prediction = classificacao(arvore, row[1])
		teste.loc[row[0], 'classified'] = prediction
	#print("final: {}".format(teste))
	return(teste, arvore)

# DIVISAO DE DADOS EM K FOLDS  - RETORNA UMA LISTA DE K PARES <TREINO, TESTE> SEGUINDO A IDEIA DA CROSS VALIDACAO


def kfold(dataset, k, seed):
    size = len(dataset)
    subset_size = round(size / k)
    dataset = shuffle(dataset)
    subsets = []
    for x in range(0, k):
        subsets.append(dataset.iloc[x*subset_size:(x*subset_size)+subset_size])
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = pd.DataFrame(columns=columns)
        for subset in subsets:
            if not test.equals(subset):
                train = train.append(subset)
        kfolds.append((train, test))

    return kfolds

# gera matriz de confusao


def get_confusion_matrix(data, expected, classified):
    classes = data[expected].unique()
    classes.sort()

    result = pd.DataFrame(columns=classes)

    for expected_class in classes:
        for classified_class in classes:
            i = data[(data[expected] == expected_class) &
                      (data[classified] == classified_class)]
            result.loc[expected_class, classified_class] = i.shape[0]

    return result

# EXECUTA VALIDACAO - CROSS VALIDACAO EM 5 FOLDS E MOSTRA AS MATRIZES E UMA MATRIZ DE MÉDIA


def cross_validation(df_treino):
	folds = kfold(df_treino, 5, 1)
	all_matrix = []
	cont = maior = 0
	for f in folds:
		treino, validacao = f
		result, arvore = decision_tree(treino, validacao, 10, 4)
		matrix = get_confusion_matrix(result, 'class', 'classified')
		if (precisao(matrix)) > maior:
			maior = precisao(matrix)
			melhor_arvore = arvore
		all_matrix.append(matrix)
		print("Fold {}:".format(cont))
		print(matrix)
		print("")
		cont+=1
	mean_matrix = pd.concat(all_matrix).groupby(level=0).mean()
	print("Mean cross validation:")
	print(mean_matrix)
	print("precision: {}".format(maior))
	return melhor_arvore

def precisao(matriz ):
	cols = list(matriz.columns)
	total = 0
	#print(matriz)
	soma = 0
	for a in matriz.index:
		soma = soma + matriz.loc[a, a]
		for c in cols:
			total = total + matriz.loc[a, c]

	return soma/total

# TESTE DE MODELO APLICANDO a "melhor" arvore
def test_model(melhor_arvore, df_teste):
    result, arvore = decision_tree_classification(melhor_arvore, df_teste)
    matrix = get_confusion_matrix(result, 'class','classified')
    print("Test model result:")
    print(matrix)
    print("")

# percorre o dataset e separa treino e teste
columns = ['sepal_length', 'sepal_width','petal_length','petal_width','class']
df_treino = pd.read_csv('iris.data', names=columns)

df_teste = pd.DataFrame(columns=columns)
for i in range(3):
    df_subset = df_treino.iloc[i*50:(i*50)+50].sample(frac=.3)
    df_teste = df_teste.append(df_subset)

df_treino = df_treino.drop(df_teste.index)

# EXECUTA A CROSS VALIDACAO
melhor_arvore = cross_validation(df_treino)

# EXECUTA TESTE DO MODELO
test_model(melhor_arvore, df_teste)
