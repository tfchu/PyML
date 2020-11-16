import tensorflow as tf

'''
tensorflow > .0
'''

# a = 2
# b = 3
# c = tf.add(a, b, name='add')
# print(c)

x = tf.constant([1,2,3,4,5])
y = tf.constant([1,1,1,1,1])
z = tf.add(x,y)
# print(z)
tf.keras.backend.print_tensor(z) 