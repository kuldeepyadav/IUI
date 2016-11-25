# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 17:24:27 2016

@author: q409NMNQ
"""

from ReadDataFromFiles import *
#from PartofSpeechExtraction import *
from NumpyArrayConversion import *
from GenerateCharEmbeddings import *
from Evaluation import *
from configuration import *


import keras
from keras.models import Sequential, Graph
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten, Merge, Permute, Reshape, Masking
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop, SGD
from keras.utils.np_utils import accuracy
from Evaluation import *


from Utilities import *
from os import walk
import os
import pickle
import numpy as np
import gc
import time


def loadDataset(): 

    picklePath = PICKLEFILEPATH
    postLabelDict = pickle.load(open(picklePath, "rb" ))
    return postLabelDict["posts"], postLabelDict["labels"]

def writeToLogFile(logString):
    f = open ("log.txt", "a")
    f.write(logString + "\n")
    f.close()


def getLinearModel(LSTM_HIDDEN_STATES=300, FIRST_DROPOUT=0.0, SECOND_DROPOUT=0.0, DENSE_LAYERS_SIZE = 300, seqLength=64, word_data_dim=100, char_data_dim=43, optimizer = 'rmsprop', lr = 0.001):

    nb_classes = 2
    nb_filters = 10

    decoder = Graph()

    decoder.add_input(name='input1', input_shape=(seqLength, word_data_dim), dtype='float')
    decoder.add_input(name='input2', input_shape=(seqLength, char_data_dim), dtype='float')

    decoder.add_node(Dropout(0.0), name='mergedInput', inputs=['input1', 'input2'])
    #decoder.add_node(Masking(mask_value=0.,), input = 'mergedInput', name='maskedInput')
    decoder.add_node(LSTM(LSTM_HIDDEN_STATES,  return_sequences = True), input= 'mergedInput', name='LSTMForward')
    decoder.add_node(Dropout(FIRST_DROPOUT), name='firstDropout', input='LSTMForward')
    decoder.add_node(TimeDistributedDense(DENSE_LAYERS_SIZE, activation='relu'), name='tdd1', input='firstDropout')
    decoder.add_node(Dropout(SECOND_DROPOUT), name = 'secondDropout', input='tdd1')
    decoder.add_node(TimeDistributedDense(nb_classes, activation='softmax'), input= 'secondDropout', name='tdd2')
    decoder.add_output(name='output', input='tdd2')

    if optimizer == 'rmsprop':
        optimizer= RMSprop(lr)
    elif optimizer == 'sgd':
        optimizer = SGD(lr)

    decoder.compile(optimizer, {'output': 'categorical_crossentropy'}, metrics=['accuracy'])

    return decoder
 

  
def getModel(LSTM_HIDDEN_STATES=300, FIRST_DROPOUT=0.0, SECOND_DROPOUT=0.0, DENSE_LAYERS_SIZE = 300, seqLength=64, word_data_dim=100, char_data_dim=43, optimizer = 'rmsprop', lr = 0.001):

    nb_classes = 2
    nb_filters = 10   
   
    decoder = Graph()

    decoder.add_input(name='input1', input_shape=(seqLength, word_data_dim), dtype='float')
    decoder.add_input(name='input2', input_shape=(seqLength, char_data_dim), dtype='float')

    decoder.add_node(Dropout(0.0), name='mergedInput', inputs=['input1', 'input2'])
    #decoder.add_node(Masking(mask_value=0.,), input = 'mergedInput', name='maskedInput')
    decoder.add_node(LSTM(LSTM_HIDDEN_STATES,  return_sequences = True), input= 'mergedInput', name='LSTMForward')
    decoder.add_node(LSTM(LSTM_HIDDEN_STATES,  return_sequences = True, go_backwards=True), input= 'mergedInput', name='LSTMBackward')
    decoder.add_node(Dropout(FIRST_DROPOUT), name='firstDropout', inputs=['LSTMForward', 'LSTMBackward'])
    decoder.add_node(TimeDistributedDense(DENSE_LAYERS_SIZE, activation='relu'), name='tdd1', input='firstDropout')
    decoder.add_node(Dropout(SECOND_DROPOUT), name = 'secondDropout', input='tdd1')
    decoder.add_node(TimeDistributedDense(nb_classes, activation='softmax'), input= 'secondDropout', name='tdd2')
    decoder.add_output(name='output', input='tdd2')

    if optimizer == 'rmsprop':
    	optimizer= RMSprop(lr)
    elif optimizer == 'sgd':
	optimizer = SGD(lr)

    decoder.compile(optimizer, {'output': 'categorical_crossentropy'}, metrics=['accuracy'])

    return decoder

def getSingleInputModel(LSTM_HIDDEN_STATES=300, FIRST_DROPOUT=0.0, SECOND_DROPOUT=0.0, DENSE_LAYERS_SIZE = 300, seqLength=64, word_data_dim=300, char_data_dim=43, optimizer = 'rmsprop', lr = 0.001):

    nb_classes = 2
    nb_filters = 10   
   
    decoder = Graph()

    decoder.add_input(name='input1', input_shape=(seqLength, word_data_dim), dtype='float')
    #decoder.add_node(Masking(mask_value=0.,), input = 'mergedInput', name='maskedInput')
    decoder.add_node(LSTM(LSTM_HIDDEN_STATES,  return_sequences = True), input= 'input1', name='LSTMForward')
    decoder.add_node(LSTM(LSTM_HIDDEN_STATES,  return_sequences = True, go_backwards=True), input= 'input1', name='LSTMBackward')
    decoder.add_node(Dropout(FIRST_DROPOUT), name='firstDropout', inputs=['LSTMForward', 'LSTMBackward'])
    decoder.add_node(TimeDistributedDense(DENSE_LAYERS_SIZE, activation='relu'), name='tdd1', input='firstDropout')
    decoder.add_node(Dropout(SECOND_DROPOUT), name = 'secondDropout', input='tdd1')
    decoder.add_node(TimeDistributedDense(nb_classes, activation='softmax'), input= 'secondDropout', name='tdd2')
    decoder.add_output(name='output', input='tdd2')

    if optimizer == 'rmsprop':
    	optimizer= RMSprop(lr)
    elif optimizer == 'sgd':
	optimizer = SGD(lr)

    decoder.compile(optimizer, {'output': 'categorical_crossentropy'}, metrics=['accuracy'])

    return decoder




postDictList, labelDictList = loadDataset()
print "Length of data : ", len(postDictList), len(labelDictList)

maxSeqLen = findBigggestSequence(labelDictList)
print "max seq len is : ", maxSeqLen

postDictList, labelDictList = shuffleLists (postDictList, labelDictList)
print "Length of data after shuffling : ", len(postDictList), len(labelDictList)

print "Training TRAIN_THRESHOLD is : ", TRAIN_THRESHOLD

trainPostDictList = postDictList[:TRAIN_THRESHOLD]
trainLabelDictList = labelDictList[:TRAIN_THRESHOLD]

testPostDictList = postDictList[TRAIN_THRESHOLD:]
testLabelDictList = labelDictList[TRAIN_THRESHOLD:]

print "Training data length : ", len(trainPostDictList), len(trainLabelDictList)
print "Testing data length : ", len(testPostDictList), len(testLabelDictList)

#newPostDictList, newLabelDictList = filterLists(postDictList, labelDictList)
#postDictList = newPostDictList
#labelDictList = newLabelDictList

#posNPArray = pickle.load(open(posNPArrayPath, "rb" ))

#totalLength = len(postDictList)


#def __init__(self, maxLen, embeddingSize, embeddingPath, isDictPickle, isCharEmbed=False, charEmbedPath=None, charEmbedSize=0)
conversionInstance = NumpyArrayConversion(maxSeqLen, wordEmbeddingSize, wordEmbeddingPath, False, False, charEmbeddingPath, charEmbeddingSize)
testPostNPArray, testLabelNPArray = conversionInstance.getNumPyArray (testPostDictList, testLabelDictList)
testX, testY = getTrainingData(testPostNPArray, testLabelNPArray)
print "Dimensions of test data : ", testX.shape, testY.shape


numEpochs = 50
batch_size = 64
maxIndex = len(trainPostDictList)

model = getSingleInputModel (300, 0.5, 0.5, 300, maxSeqLen)

print "Model compileds, starting training with epochs and batch size : ", numEpochs, batch_size

for epochIndex in range(numEpochs):
    startIndex = 0
    lastIndex = startIndex + batch_size 
    startTime = time.time()
    while (lastIndex <= maxIndex):
        postNPArray, labelNPArray = conversionInstance.getNumPyArray (trainPostDictList[startIndex:lastIndex], trainLabelDictList[startIndex:lastIndex])
        #charNPArray = conversionInstance.getCharNumPyArray(postDictList[startIndex:lastIndex], charEmbeddingSize)
        #print "Shape of postNP Array and label NP Array : ", postNPArray.shape, labelNPArray.shape
        #print "Shape of charNPArray : ", charNPArray.shape

        bTrainX, bTrainY = getTrainingData(postNPArray, labelNPArray)
        #print "Dimensions of TrainX and Train Y : ", trainX.shape, trainY.shape

        #newTrainCharX = np.concatenate((charNPArray, posNPArray), axis=2)
        #print "Dimensions of newTrainCharX = ", newTrainCharX.shape

        model.train_on_batch({'input1' : bTrainX, 'output': bTrainY})
        startIndex = lastIndex
	lastIndex = startIndex + batch_size

	if lastIndex % 50032 == 0:
	   print "One l rows completed ", time.time() - startTime
	   startTime = time.time()

    print "start and last index : ", startIndex, lastIndex
    y_pred = model.predict({'input1' : testX})
    f_score, precision, recall  = evaluateModel(y_pred, testY)        
    print "Evaluation : ", epochIndex, f_score, precision, recall
    with open("result_log.txt", "a") as f:
        f.write(str(epochIndex) + "," + str(f_score) + "," + str(precision) + "," + str(recall) + "\n") 

    json_string = model.to_json()
    with open(MODELFOLDERNAME + "model_specifications.json", "w") as f:
         f.write (json_string)

    model.save_weights(MODELFOLDERNAME + 'allBooksKerasWeights_' + str(epochIndex) + '.h5')
    model.save(MODELFOLDERNAME + 'allBooksKerasModel_' + str(epochIndex) + '.h5')


trainPostDictList, trainLabelDictList = shuffleLists (trainPostDictList, trainLabelDictList)
print "Length of data after shuffling : ", len(trainPostDictList), len(trainLabelDictList)

model.save(MODELFOLDERNAME + 'allBooks_Keras_model.h5')
json_string = model.to_json()
with open(MODELFOLDERNAME + "model_specification.json", "w") as f:
     f.write (json_string)

model.save_weights(MODELFOLDERNAME + 'allBooks_Keras_weights.h5')
#model.save('allBooks_Keras_model.h5')



print "All epoch finished"
