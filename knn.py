
from __future__ import print_function

import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
# import matplotlib.pyplot as plt
localtime = time.asctime( time.localtime(time.time()) )
print("本地时间为 :", localtime)
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# pick 2000 samples to speed up testing

train_data = torchvision.datasets.MNIST(root='./mnist/', train=True)
train_x = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor)[:60000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
train_y = train_data.train_labels[:60000]

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]

print(train_x.size(),train_y.size(),test_x.size(),test_y.size())

train_x = train_x.view(-1,28*28)
test_x = test_x.view(-1,28*28)





# K-Nearest Neighbor Classification

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
# import matplotlib.pyplot as plt
import numpy as np
# import imutils
# import cv2

# load the MNIST digits dataset
# mnist = datasets.load_digits()
# print(len(np.array(mnist.data)[0]))


# Training and testing split,
# 75% for training and 25% for testing
# print(len(mnist))
# (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

# take 10% of the training data and use that for validation
# (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)
trainData = np.array(train_x)
testData = np.array(test_x)
trainLabels = np.array(train_y)
testLabels = np.array(test_y)
valData = testData
valLabels = testLabels

# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))


# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over kVals
for k in range(1, 30, 2):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
    localtime = time.asctime( time.localtime(time.time()) )
    print("本地时间为 :", localtime)

# largest accuracy
# np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))


# Now that I know the best value of k, re-train the classifier
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)



# Predict labels for the test set
predictions = model.predict(testData)

# Evaluate performance of model for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

# some indices are classified correctly 100% of the time (precision = 1)
# high accuracy (98%)

# check predictions against images
# loop over a few random digits
image = testData
j = 0
for i in np.random.randint(0, high=len(testLabels), size=(24,)):
        # np.random.randint(low, high=None, size=None, dtype='l')
    prediction = model.predict(image)[i]
    image0 = image[i].reshape((8, 8)).astype("uint8")
    image0 = exposure.rescale_intensity(image0, out_range=(0, 255))
    # plt.subplot(4,6,j+1)
    # plt.title(str(prediction))
    # plt.imshow(image0,cmap='gray')
    # plt.axis('off')


        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels for better visualization

        #image0 = imutils.resize(image[0], width=32, inter=cv2.INTER_CUBIC)

    j = j+1

    # show the prediction
    # print("I think that digit is: {}".format(prediction))
    # print('image0 is ',image0)
    # cv2.imshow("Image", image0)
    # cv2.waitKey(0) # press enter to view each one!
# plt.show()