from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

x = tf.constant([[1],[2],[3],[4]], dtype = tf.float32)
yTrue  = tf.constant([[0],[-2],[-1],[-3]] , dtype =tf.float32)

linearModel = tf.layers.Dense(units= 1)

yPred = linearModel(x)

sess = tf.Session()
init  = tf.global_variables_initializer()
sess.run(init)

print(sess.run(yPred))

loss = tf.losses.mean_squared_error(labels = yTrue , predictions = yPred)

print(sess.run(loss))

oprimizer = tf.train.GradientDescentOptimizer(0.01)
train = oprimizer.minimize(loss)

for i in range(100):
	_,lossValue = sess.run((train,loss))
	print(lossValue)