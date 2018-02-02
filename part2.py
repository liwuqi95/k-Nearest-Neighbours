import part1 as part1
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)

trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]



sess = tf.Session()
def getR(D, k):
	topK = tf.nn.top_k(-D,k)
	oneHot = tf.one_hot(topK.indices, tf.shape(D)[1], on_value=1/k, off_value=0.0, axis=-1)
	return tf.cast(tf.reduce_sum(oneHot,1), tf.float64)




def predict(newData, trainData, trainTarget, k):
	dis = part1.pairwiseDistance(newData, trainData)
	r = getR(dis, k)
	return tf.matmul(r, trainTarget)


def MSE(predY, newY):
	return tf.reduce_mean(tf.reduce_sum(pow((predY - newY),2), 1))


kv = [1,3,5,50]


X = np.linspace(0.0,11.0,num=1000)[:,np.newaxis] 


for k in kv:

	trainMSE  = MSE(predict( trainData,trainData,trainTarget,k),trainTarget)
	validMSE = MSE(predict( validData,trainData,trainTarget,k),validTarget)
	testMSE  = MSE(predict( testData,trainData,trainTarget,k),testTarget)

	result = predict(X, trainData, trainTarget, k)

	print('With k = ' + str(k) + "   trainMSE = " + str(sess.run(trainMSE)) + "   validMSE = " + str(sess.run(validMSE))+"   testMSE = " + str(sess.run(testMSE)))


	plt.figure(k)
	plt.plot(trainData,trainTarget,'.')



	plt.plot(X,sess.run(result),'-')
	plt.title("k-NN regression on datat1D, k=%d"%k)
	plt.show()














