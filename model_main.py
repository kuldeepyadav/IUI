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


def loadDataset(): 

    picklePath = DATABASEPATH + "postLabelDict.pickle"
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

postDictList, labelDictList = shuffleLists (postDictList, labelDictList)

#newPostDictList, newLabelDictList = filterLists(postDictList, labelDictList)
#postDictList = newPostDictList
#labelDictList = newLabelDictList

print "Length of data after shuffling : ", len(postDictList), len(labelDictList)

#posNPArray = pickle.load(open(posNPArrayPath, "rb" ))


#totalLength = len(postDictList)
maxSeqLen = findBigggestSequence(labelDictList)
print "max seq len is : ", maxSeqLen

#def __init__(self, maxLen, embeddingSize, embeddingPath, isDictPickle, isCharEmbed=False, charEmbedPath=None, charEmbedSize=0)
conversionInstance = NumpyArrayConversion(maxSeqLen, wordEmbeddingSize, wordEmbeddingPath, False, False, charEmbeddingPath, charEmbeddingSize)


startIndex = 0
lastIndex = len(postDictList)
#model = getModel(500, 0.1, 0.4, 500, maxSeqLen, wordEmbeddingSize, charEmbeddingSize)
#self, maxLen, embeddingSize, embeddingPath, isDictPickle, isCharEmbed=False, charEmbedPath=None, charEmbedSize=0)                
postNPArray, labelNPArray = conversionInstance.getNumPyArray (postDictList[startIndex:lastIndex], labelDictList[startIndex:lastIndex])
#charNPArray = conversionInstance.getCharNumPyArray(postDictList[startIndex:lastIndex], charEmbeddingSize)
print "Shape of postNP Array and label NP Array : ", postNPArray.shape, labelNPArray.shape
#print "Shape of charNPArray : ", charNPArray.shape

trainX, trainY = getTrainingData(postNPArray, labelNPArray)
print "Dimensions of TrainX and Train Y : ", trainX.shape, trainY.shape

#newTrainCharX = np.concatenate((charNPArray, posNPArray), axis=2)
#print "Dimensions of newTrainCharX = ", newTrainCharX.shape


#posNPArray = getNPArrayOfSentenceFeatures(postDictList[startIndex:lastIndex], maxSeqLen)
#print "posNPArray shape : ", posNPArray.shape
#print postNPArray[0]
#print labelNPArray[1]
#writeToLogFile("Post array sample : " + str( postNPArray[0]))
#writeToLogFile( "Label array sample : " + str(labelNPArray[0]))
#writeToLogFile("POS array sample : " + str( posNPArray[0]))

#trainX, trainY = getTrainingData(postNPArray, labelNPArray)
#newTrainCharX = np.concatenate((charNPArray, posNPArray), axis=2)
#print "Dimensions of new#newTrainCharX = np.concatenate((charNPArray, posNPArray), axis=2)
#print "Dimensions of new TrainCharX : ", newTrainCharX.shape TrainCharX : ", newTrainCharX.shape

class ComputeMetrics(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.metricsDict={}
	return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
       
        y_pred = self.model.predict({'input1' : trainX[TRAIN_THRESHOLD:]})
        f_score, precision, recall  = evaluateModel(y_pred, trainY[TRAIN_THRESHOLD:])        
        print "Evaluation : ", epoch, f_score, precision, recall
	self.metricsDict[epoch] = f_score, precision, recall

	with open("result_log.txt", "a") as f:
             f.write(str(epoch) + "," + str(f_score) + "," + str(precision) + "," + str(recall) + "\n")	     
     
	return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

    def getMetricsDict(self):
        return self.metricsDict

    def getMaximums(self):
        f_score = 0
        precision = 0
        recall = 0

        for key, value in self.metricsDict.items():
            if float(value[0]) > f_score:
               f_score = float(value[0])
	       precision = float(value[1])
	       recall = float(value[2])

	return f_score, precision, recall



evaluation = ComputeMetrics()
model = getSingleInputModel (300, 0.0, 0.0, 300, maxSeqLen)
model.fit({'input1' : trainX[:TRAIN_THRESHOLD], 'output': trainY[:TRAIN_THRESHOLD]}, batch_size=64, nb_epoch=50, show_accuracy=True, callbacks=[evaluation])
f_score, precision, recall  = evaluation.getMaximums()
print f_score, precision, recall

#evaluation = ComputeMetrics()
#model = getSingleInputModel (300, 0.2, 0.3, 300, maxSeqLen)
#model.fit({'input1' : trainX[:TRAIN_THRESHOLD], 'output': trainY[:TRAIN_THRESHOLD]}, batch_size=64, nb_epoch=50, show_accuracy=True, callbacks=[evaluation])
#f_score, precision, recall  = evaluation.getMaximums()
#print f_score, precision, recall

#accuracy  = model.evaluate({'input1' : trainX[4500:], 'input2' : newTrainCharX[4500:], 'output': trainY[4500:]})
#print accuracy

"""
#Varying hidden layers and dropouts

hiddenlayers = [50,100,150,200,250,300,350,400,450,500]
dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for hiddenlayer in hiddenlayers:
    for firstdropout in dropouts:
        for seconddropout in dropouts:
	    evaluation = ComputeMetrics()
            model = getModel(hiddenlayer, firstdropout, seconddropout, hiddenlayer, maxSeqLen)
	    model.fit({'input1' : trainX[:4000], 'input2' : newTrainCharX[:4000], 'output': trainY[:4000]}, batch_size=64, nb_epoch=50, show_accuracy=True, callbacks=[evaluation])                	
            f_score, precision, recall  = evaluation.getMaximums()
	    with open("summaryResults.txt", "a") as f:
		f.write(str(hiddenlayer)+ "," + str(firstdropout) + "," +str(seconddropout) + "," + str(f_score) + "," +str(precision) + "," + str(recall))
            


for hiddenlayer in hiddenlayers:
    for firstdropout in dropouts:
        for seconddropout in dropouts:
            evaluation = ComputeMetrics()
            model = getLinearModel(hiddenlayer, firstdropout, seconddropout, hiddenlayer, maxSeqLen)
            model.fit({'input1' : trainX[:4000], 'input2' : newTrainCharX[:4000], 'output': trainY[:4000]}, batch_size=64, nb_epoch=50, show_accuracy=True, callbacks=[evaluation])
            f_score, precision, recall  = evaluation.getMaximums()
            with open("summaryResults.txt", "a") as f:
                f.write(str(hiddenlayer)+ "," + str(firstdropout) + "," +str(seconddropout) + "," + str(f_score) + "," +str(precision) + "," + str(recall))

lrs = [0.1,0.01,0.015,0.001,0.0015, 0.0001]

for lr in lrs:

    evaluation = ComputeMetrics()
    model = getLinearModel(100, 0.1, 0.4, 100, maxSeqLen, 100, 43, 'sgd', lr)
    model.fit({'input1' : trainX[:4000], 'input2' : newTrainCharX[:4000], 'output': trainY[:4000]}, batch_size=64,                nb_epoch=50, show_accuracy=True, callbacks=[evaluation])
    f_score, precision, recall  = evaluation.getMaximums()
    with open("summaryResults.txt", "a") as f:
         f.write(str('sgd') + "," + str(lr) + "," + str(f_score) + "," +str(precision) + "," + str(recall))
 
    evaluation = ComputeMetrics()
    model = getLinearModel(100, 0.1, 0.4, 100, maxSeqLen, 100, 43, 'rmsprop', lr)
    model.fit({'input1' : trainX[:4000], 'input2' : newTrainCharX[:4000], 'output': trainY[:4000]}, batch_size=64,                nb_epoch=50, show_accuracy=True, callbacks=[evaluation])
    f_score, precision, recall  = evaluation.getMaximums()
    with open("summaryResults.txt", "a") as f:
         f.write(str('rmsprop') + "," + str(lr) + "," + str(f_score) + "," +str(precision) + "," + str(recall))    
"""
        
