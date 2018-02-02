


import tensorflow as tf
import part1 as part1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


sess = tf.Session()

def data_segmentation(data_path, target_path, task):

# task = 0 >> select the name ID targets for face recognition task
# task = 1 >> select the gender ID targets for gender recognition task data = np.load(data_path)/255
	data = np.load(data_path)/255
	data = np.reshape(data, [-1, 32*32])
	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
	data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
	data[rnd_idx[trBatch + validBatch+1:-1],:]
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
	target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
	target[rnd_idx[trBatch + validBatch + 1:-1], task]

	return trainData, validData, testData, trainTarget, validTarget, testTarget


def getR(D, k):
	topK = tf.nn.top_k(-D,k)
	oneHot = tf.one_hot(topK.indices, tf.shape(D)[1], on_value=1/k, off_value=0.0, axis=-1)
	return tf.cast(tf.reduce_sum(oneHot,1), tf.float64)


	# get the major vote
def getMajor(r, trainTarget):
	votes = tf.boolean_mask(trainTarget, tf.cast(r, tf.bool))
	y, idx, count = tf.unique_with_counts(votes)
	return y[tf.argmax(count)]

def accuracy(predY, newY):
	return tf.reduce_mean(tf.cast(tf.equal(predY,newY),tf.float64))


# predict by major voting
def predictMV(newData, trainData, trainTarget, k):

	dis = part1.pairwiseDistance(newData, trainData)
	r = getR(dis, k) * k
	return tf.map_fn(lambda x: getMajor(x, tf.cast(trainTarget,tf.float64)), r)
	
trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("./data.npy","./target.npy", 1)



def display(data):
	data = tf.reshape(data, [32,32])
	imgplot = plt.imshow(sess.run(data), cmap = 'gray')
	plt.show()
	return 1


kv = [1,5,10,25,50,100,200]

validAC_list = []

# find the k that highest accuracy with validation test
for k in kv:

    # get the validation accuracy
	validAC = accuracy(predictMV(validData,trainData,trainTarget,k),validTarget)

	validAC_list.append(sess.run(validAC))

	print('With k = ' + str(k) + "   valid accuracy = " + str(sess.run(validAC)))



bestK = kv[np.argmax(validAC_list)]

print('The best k (with highest accuracy is) ' + str(bestK))



testAC = accuracy(predictMV(testData,trainData,trainTarget,bestK),testTarget)

print('The best k running testData get accuracy = ' + str(sess.run(testAC)))



k = 10

result = predictMV(testData,trainData,trainTarget,k)

index = tf.where(tf.not_equal(result, testTarget))[0][0]

dis = part1.pairwiseDistance(testData, trainData)

r = (getR(dis, k) * k)[index]

knnData = tf.boolean_mask(trainData, tf.cast(r, tf.bool))

knnVotes = tf.boolean_mask(trainTarget, tf.cast(r, tf.bool))

print('Votes are:')
print(sess.run(knnVotes))
print('actul result is ' + str(sess.run(index)))

for i in range(0,k):
	display(knnData[i])

display(tf.cast(testData,tf.float64)[index])







