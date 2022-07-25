from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical



# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()

	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)

	return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')

	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
    
	# return normalized images
	return train_norm, test_norm


if __name__ == "__main__":
    # load dataset
    trainX, trainy, testX, testy = load_dataset()

	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)

    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(trainX[i])

    # show the figure
    pyplot.show()
