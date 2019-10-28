from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import xlrd
import math

path = 'D:/Projects/Git2/Tensorflow_core_gym-/dataset2.xlsx'

def load_data(PATH):
	# loading data from exel
	database1 = xlrd.open_workbook(PATH)
	sheet = database1.sheet_by_index(0)
	fileds = []

	m = sheet.nrows 
	m_traning = math.floor(m*0.75)
	m_testing = math.floor(m*0.25)


	training_set = np.zeros((m_traning,sheet.ncols))
	testing_set = np.zeros((m_testing,sheet.ncols))

	for i in range(0,m_traning):
		for j in range(0,sheet.ncols):
			try:
				training_set[i][j] = sheet.cell_value(i,j)
			except:
				fileds.append(sheet.cell_value(i,j))
	p=0;

	for i in range(m_traning,m):
		p = p + 1
		for j in range(0,sheet.ncols):
			try:
				testing_set[p][j] = sheet.cell_value(i,j)
			except:
				fileds.append(sheet.cell_value(i,j))

	training_set = np.delete(training_set,0,0) #Removing the top row
	testing_set = np.delete(testing_set,0,0)  #Removing the top row

	return training_set,testing_set

def separateInputAndOutput(train,test):
	nColumns = train.shape[1]
	columnsToBeDeleted = np.arange(nColumns-1)

	lastColumnDeleted_train = np.delete(train, nColumns - 1 , 1)
	lastColumnDeleted_test = np.delete(test,nColumns - 1, 1)
	firstColumsDeleted_train = np.delete(train, columnsToBeDeleted , 1)
	firstColumsDeleted_test = np.delete(test, columnsToBeDeleted , 1)

	return np.transpose(lastColumnDeleted_train),np.transpose(firstColumsDeleted_train),np.transpose(lastColumnDeleted_test),np.transpose(firstColumsDeleted_test)
	
def model(x,w1,b1):
	return tf.add(tf.matmul(w1,x),b1)


train, test = load_data(path)
xTrain, yTrain, xTest, yTest = separateInputAndOutput(train,test)

nTrain = xTrain.shape[1]
nTest = xTest.shape[1] 

print('xTrain:',xTrain.shape,' yTrain:',yTrain.shape,' xTest:',xTest.shape,' yTest:',yTest.shape)
print('Train Samples:',nTrain,' Test Samples',nTest)

w1 = tf.Variable(tf.zeros([3,2]), dtype = tf.float64) #[nInputs,nOutputs] <-- from the layer
w2 = tf.Variable(tf.zeros([2,2]))
w3 = tf.Variable(tf.zeros([2,1]))

b1 = tf.Variable(tf.zeros([2,nTrain])) # noutputs from the layer
b2 = tf.Variable(tf.zeros([2,nTrain]))
b3 = tf.Variable(tf.zeros([1,nTrain]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(model(xTrain,w1,b1)))



