import tensorflow as tf


def pairwiseDistance(X, Z):

   #expend X
   xE = tf.expand_dims(X,2)

   #expend Z
   zE = tf.expand_dims(tf.transpose(Z),0)

   #get power of 2
   p = tf.pow((xE - zE),2)
   
   return tf.reduce_sum(p, 1)







