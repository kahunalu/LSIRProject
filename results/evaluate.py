import itertools
from scipy.stats import ks_2samp
import numpy as np

def evalKS(obs, exp):
	obs_array=[]
	exp_array=[]

	for i in range(0,10):
		obs_temp=[]
		exp_temp=[]
		for j in range(0,10):
			obs_value = obs[(i,j)]
			exp_value = exp[(i,j)]

			obs_temp.append(obs_value)
			exp_temp.append(exp_value)

			obs_array.append(obs_value)
			exp_array.append(exp_value)

	print ks_2samp(np.array(obs_array), np.array(exp_array))


def constructMatrix(matrix, predictions_set, test_set):
	for i, prediction in enumerate(predictions_set):
		matrix[(prediction, test_set[i])] = matrix[(prediction, test_set[i])] + 1

	return matrix


def findsubsets(S):
	return set(itertools.product(S, repeat=2))

category_list = [0,1,2,3,4,5,6,7,8,9]

category_list = findsubsets(category_list)

used_matrix = {}
imagenet_matrix = {}
imagenet_2_matrix = {}
for i in category_list:
	used_matrix[i] = 0
	imagenet_matrix[i] = 0
	imagenet_2_matrix[i] = 0

used_test = np.reshape(np.load("used_test.dat"), (1400))
used_predictions = np.reshape(np.load("used_predictions.dat"), (1400))

used_matrix = constructMatrix(used_matrix, used_predictions, used_test)

imagenet_test = np.reshape(np.load("imagenet_test.dat"), (1400))
imagenet_predictions = np.reshape(np.load("imagenet_predictions.dat"), (1400))

imagenet_matrix = constructMatrix(imagenet_matrix, imagenet_predictions, imagenet_test)

imagenet_test2 = np.reshape(np.load("imagenet_test2.dat"), (1400))
imagenet_predictions2 = np.reshape(np.load("imagenet_predictions2.dat"), (1400))

imagenet_2_matrix = constructMatrix(imagenet_2_matrix, imagenet_predictions2, imagenet_test2)

print "Comparing imagenet to imagenet"
#compare imagenet_2_matrix to imagenet_matrix results
evalKS(obs=imagenet_2_matrix, exp=imagenet_matrix)


print "Comparing used to imagenet"
#compare used_matrix to imagenet_matrix results
#evalChiSquare(obs=used_matrix, exp=imagenet_matrix)


