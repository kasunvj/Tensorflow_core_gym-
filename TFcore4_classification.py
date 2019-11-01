
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


import numpy as np
import tensorflow as tf
import xlrd
import math
import pandas as pd

# >       X       X       X  >
# >  w1   X  w2   X  w3
# >     


path = 'D:/Projects/Git2/Tensorflow_core_gym-/dataset2.csv'

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

	return lastColumnDeleted_train,firstColumsDeleted_train,lastColumnDeleted_test,firstColumsDeleted_test
	
# #train, test = load_data(path)
# #xTrain, yTrain, xTest, yTest = separateInputAndOutput(train,test)

# nTrain = xTrain.shape[1]
# nTest = xTest.shape[1] 

# print("-----------------------------------------")
# print('xTrain:',xTrain.shape,'\nyTrain:',yTrain.shape,'\nxTest:',xTest.shape,'\nyTest:',yTest.shape)
# print('Train Samples:',nTrain,'\nTest Samples',nTest)
# print("-----------------------------------------")

df = pd.read_csv(path)
#print(df.head())

properties = list(df.columns.values)
properties.remove('y')
print(properties)

x = df[properties]
y = df['y']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

model = keras.Sequential([
	keras.layers.Flatten(input_shape = (3,)),
	keras.layers.Dense(2, activation = tf.nn.relu),
	keras.layers.Dense(2, activation = tf.nn.relu),
	keras.layers.Dense(1, activation = tf.nn.sigmoid)
	])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs = 20 , batch_size = 1)
test_loss , test_acc = model.evaluate(xTest,yTest)
print('Test acc:',test_acc)

a = np.array([[4,4,6]])
print(model.predict(a))