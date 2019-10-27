#https://www.tensorflow.org/guide/low_level_intro
#Glosary 
#tensor: tensor consists of a set of primitive values shaped into an array of any number of dimension
#rank: is its number of dimensions
#shape: tuple with integers specifying the array's length along each dimension
#computational graph: is a series of TensorFlow operations arranged into a graph
# placeholder: is a promise to provide a value later, like a function argument

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf 

a = tf.constant(3.0 , dtype = tf.float32)
b = tf.constant(4.0)
total = a+b
print(a)  #Tensor("Const:0", shape=(), dtype=float32)
print(b)  #Tensor("Const_1:0", shape=(), dtype=float32)
print(total)  #Tensor("add:0", shape=(), dtype=float32)

'''Notice that printing the tensors does not output the values
 3.0, 4.0, and 7.0 as you might expect. 
 The above statements only build the computation graph. 
 These tf.Tensor objects just represent the results of the 
 operations that will be run.'''

#Tensor board
#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())
#writer.flush()

sess = tf.Session()
print(sess.run(total)) # 7

print(sess.run({'total: ':total , 'a+3*b: ':a +3*b, 'a: ':a, 'b: ':b}))
#{'total': 7.0, 'ab': (3.0, 4.0), 'a': 3.0, 'b': 4.0}

vec = tf.random_uniform(shape = (3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1,out2))) 
'''[0.27247846 0.41334188 0.6734953 ]
   [0.9871659  0.8230206  0.90415084]
(array([1.0821799, 1.5444727, 1.4507784], dtype=float32), 
 array([2.08218  , 2.5444727, 2.4507785], dtype=float32))'''

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z , feed_dict = {x:3,     y:4.5}))  # 7.5
print(sess.run(z , feed_dict = {x:[1,2], y:[2,4]})) # [3,6]

# DATA_SETS____________________________________________________________________

my_data = [
	[0,1],
	[1,2],
	[2,3],
	[3,4],
]

'''Placeholders work for simple experiments, but tf.data are the preferred 
	method of streaming data into a model.
	To get a runnable tf.Tensor from a Dataset you must 
	first convert it to a tf.data.Iterator,
 	and then call the Iterator's tf.data.Iterator.get_next method.'''

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
	try:
		print(sess.run(next_item))
	except:
		break

#[0 1]
#[1 2]
#[2 3]
#[3 4]

r = tf.random_normal([10,3])
print(sess.run(r))

'''[[-1.1511577  -0.48421392  0.2115877 ]
 [ 0.2736128  -0.40367258  0.9253847 ]
 [-0.79377687  0.01603176  0.9661043 ]
 [ 0.41331738 -0.68535     0.07387666]
 [-1.3349615   1.6902134   0.56192905]
 [-1.0143543   2.0423958   0.05126164]
 [ 0.18449089  0.25637218 -1.2111031 ]
 [ 0.11218715  0.01442587 -0.55858654]
 [-0.94006866  0.7207023   0.94638914]
 [-0.34524864  0.17363387 -1.8558768 ]]'''

dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)

while True:
 	try:
 		print(sess.run(next_row))
 	except:
 		break

'''[2.0264854 1.2323453 0.3457349]
[1.682757   0.22062112 0.6691639 ]
[ 1.6432762  -0.33827618 -0.62751615]
[-0.4378814  -0.44390854  0.8421243 ]
[ 0.42275614 -0.0795733   0.74974686]
[-0.3231588 -0.7818212  1.220891 ]
[-0.5756415  -0.30553877 -0.30573046]
[ 0.43028557 -0.00630576 -0.47644565]
[ 0.1055615   0.78775203 -0.9759877 ]
[ 0.58226347 -1.6429812   0.42872325]'''


# LAYERS____________________________________________________
print("_____________________________________________________")

x = tf.placeholder(tf.float32, shape = [None, 3])
linear_model = tf.layers.Dense(units = 1)
y = linear_model(x)

'''The layer inspects its input to determine sizes for 
   its internal variables. So here we must set the shape
   of the x placeholder so that the layer can build a 
   weight matrix of the correct size.'

 The layer contains variables that must be initialized
   before they can be used. While it is possible to initialize 
   variables individually, you can easily initialize all 
   the variables in a TensorFlow graph as follows:

  Also note that this global_variables_initializer only 
   initializes variables that existed in the graph when 
   the initializer was created. So the initializer should
   be one of the last things added during graph construction. '''

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1,2,3],[2,3,4]]}))