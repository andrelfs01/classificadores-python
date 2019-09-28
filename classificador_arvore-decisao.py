import pandas as pd
from sklearn.utils import shuffle
import numpy as np


# Realiza a divisao de um corte dado, feature e valor
def testa_corte(feature, valor, dataset):
	left, right = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
	for row in dataset.iterrows():
		#print("teste {} < {}".format(row[1][feature], valor))
		if row[1][feature] < valor:
			left.loc[row[0]] = row[1]
		else:
			#right.append(row)
			right.loc[row[0]] = row[1]
	return left, right

# Calculate the Gini index for a split dataset
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
			p = [row[1][-1] for row in grupo].count(class_val) / size
			score += p * p
		# ponderando o indice pelo tamenho do grupo
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
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
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def corte(no, max_depth, min_size, depth):
	left, right = no['groups']
	del(no['groups'])

	print("corte")
	# check for a no split
	if not left.empty or not right.empty:
		no['left'] = no['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		no['left'], no['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		no['left'] = to_terminal(left)
	else:
		no['left'] = seleciona_corte(left)
		corte(no['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		no['right'] = to_terminal(right)
	else:
		no['right'] = seleciona_corte(right)
		corte(no['right'], max_depth, min_size, depth+1)

# gera a decision tree
def gera_arvore(treino, max_depth, min_size):
	raiz = seleciona_corte(treino)
	corte(raiz, max_depth, min_size, 1)
	return raiz

# Make a prediction with a decision tree
def predict(node, row):
	print("grupos {}".format(node['right']))
	print("{} - {}".format(node, row))
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Contrutor da arvore de decisao
def decision_tree(treino, teste, max_depth, min_size):
	arvore = gera_arvore(treino, max_depth, min_size)
	predictions = list()
	print("arvore {}".format(arvore))
	exit
	for row in teste:
		prediction = predict(arvore, row)
		predictions.append(prediction)
	return(predictions, arvore)

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
        result, arvore = decision_tree(treino, validacao, 5, 10)
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
    result = decision_tree(sumarizado, df_teste, 5, 10)
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