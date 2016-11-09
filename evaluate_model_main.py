# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 17:24:27 2016

@author: q409NMNQ
"""

from ReadDataFromFiles import *
from PartofSpeechExtraction import *
from NumpyArrayConversion import *
from GenerateCharEmbeddings import *
from Evaluation import *

from keras.models import Sequential, Graph
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten, Merge, Permute, Reshape
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.utils.np_utils import accuracy
from keras.models import model_from_json



from Utilities import *
from os import walk
import os
import pickle
import numpy as np
import gc


PATH = "/home/kuldeep/entitydetection/IUI/Dataset/"

#PATH = "D:\\udacity_2\\tensorflow-udacity-vagrant\\assignments\\stackoverflow\\booktext\\"
wordEmbeddingPath = PATH + "bookEmbeddingModel.bin"
charEmbeddingPath = PATH + "charEmbedPath.pickle"
wordEmbeddingSize = 300
charEmbeddingSize = 175
maxSeqLen = 200
maxWordLen = 70


def loadDataset(): 

    picklePath = PATH + "postLabelDict.pickle"
    postLabelDict = pickle.load(open(picklePath, "rb" ))
    return postLabelDict["posts"], postLabelDict["labels"]

def writeToLogFile(logString):
    f = open ("log.txt", "a")
    f.write(logString + "\n")
    f.close()

def getModel(LSTM_HIDDEN_STATES=300, FIRST_DROPOUT=0.6, SECOND_DROPOUT=0.6, DENSE_LAYERS_SIZE = 300, seqLength=64, word_data_dim=100, char_data_dim=50):

    nb_classes = 2
    nb_filters = 10   
   
    decoder = Graph()

    decoder.add_input(name='input1', input_shape=(seqLength, word_data_dim), dtype='float')
    decoder.add_input(name='input2', input_shape=(seqLength, char_data_dim), dtype='float')


    #decoder.add_node(Embedding(char_vocab_size, char_embedding_size, input_length=seqLength*max_charlen), input='input2', name='charEmbeddings')
    #decoder.add_node(Reshape((seqLength, max_charlen, char_embedding_size)), input='charEmbeddings', name='reshape')
    #decoder.add_node(Permute((3,1,2)), input='reshape', name= 'permute')
    #decoder.add_node(Convolution2D(nb_filters, 3, 3, border_mode='same'), input='permute', name='convolution2D')
    #decoder.add_node(Permute((2,1,3)), input='convolution2D', name='permute2')
    #decoder.add_node(MaxPooling2D((2, 2)), input='permute2', name='maxpool')
    #decoder.add_node(Reshape((seqLength, 40)), input='maxpool', name = 'model_cnn_output')

    decoder.add_node(Dropout(0.0), name='mergedInput', inputs=['input1', 'input2'])
    decoder.add_node(LSTM(LSTM_HIDDEN_STATES,  return_sequences = True), input= 'mergedInput', name='LSTMForward')
    decoder.add_node(LSTM(LSTM_HIDDEN_STATES,  return_sequences = True, go_backwards=True), input= 'mergedInput', name='LSTMBackward')
    decoder.add_node(Dropout(FIRST_DROPOUT), name='firstDropout', inputs=['LSTMForward', 'LSTMBackward'])
    decoder.add_node(TimeDistributedDense(DENSE_LAYERS_SIZE, activation='relu'), name='tdd1', input='firstDropout')
    decoder.add_node(Dropout(SECOND_DROPOUT), name = 'secondDropout', input='tdd1')
    decoder.add_node(TimeDistributedDense(nb_classes, activation='softmax'), input= 'secondDropout', name='tdd2')
    decoder.add_output(name='output', input='tdd2')

    optimizer='rmsprop'
    decoder.compile(optimizer, {'output': 'categorical_crossentropy'}, metrics=['accuracy'])

    return decoder



postDictList, labelDictList = loadDataset()
print "Length of data : ", len(postDictList), len(labelDictList)


newPostDictList, newLabelDictList = filterLists(postDictList, labelDictList)

postDictList = newPostDictList
labelDictList = newLabelDictList

print "Length of data : ", len(postDictList), len(labelDictList)

writeToLogFile("Post itemt : " + str( postDictList[0]))
writeToLogFile("Label item : " + str(labelDictList[0]))

#poSpeechDictList = getNormalizedSentenceFeatures(postDictList)

#picklePath = PATH + "poSpeechDict.pickle"
#pickle.dump(poSpeechDictList, open(picklePath, "wb" ))

#totalLength = len(postDictList)
totalLength = 60048
batch_size = 64
numEpochs = 20

#conversionInstance = NumpyArrayConversion(maxSeqLen, wordEmbeddingSize, wordEmbeddingPath, False)
conversionInstance = NumpyArrayConversion(maxSeqLen, wordEmbeddingSize, wordEmbeddingPath, False, True, charEmbeddingPath, charEmbeddingSize)

#model = getModel(300, 0.6, 0.6, 300, maxSeqLen, wordEmbeddingSize, charEmbeddingSize)

print "Loading model :"

MODELPATH = PATH + 'keras_model_' + str(19) + '.json'
MODELWEIGHTS = PATH + 'keras_model_' + str(19) + '.h5'

jsonModel = ''
with open(MODELPATH, 'r') as myfile:
    jsonModel = myfile.read().replace('\n', '')

#model = model_from_json(jsonModel)
#model.compile()
model = getModel(300, 0.6, 0.6, 300, maxSeqLen, wordEmbeddingSize, charEmbeddingSize)
model.load_weights(MODELWEIGHTS)

print "Model loaded with weights"

allAccuracy = []

startIndex = 50048
lastIndex = startIndex + batch_size
while (lastIndex <= totalLength):		
	
	postNPArray, labelNPArray = conversionInstance.getNumPyArray (postDictList[startIndex:lastIndex], labelDictList[startIndex:lastIndex])
	charNPArray = conversionInstance.getCharNumPyArray(postDictList[startIndex:lastIndex], charEmbeddingSize)

	print "Shape of postNP Array and label NP Array : ", postNPArray.shape, labelNPArray.shape
	print "Shape of charNPArray : ", charNPArray.shape

	#posNPArray = getNPArrayOfSentenceFeatures(postDictList[startIndex:lastIndex], maxSeqLen)

	#print "posNPArray shape : ", posNPArray.shape

	writeToLogFile("Post array sample : " + str( postNPArray[0]))
	writeToLogFile( "Label array sample : " + str(labelNPArray[0]))
	#writeToLogFile("POS array sample : " + str( posNPArray[0]))

	trainX, trainY = getTrainingData(postNPArray, labelNPArray)
	#newTrainCharX = np.concatenate((charNPArray, posNPArray), axis=2)
		
	#print "Dimensions of new TrainCharX : ", newTrainCharX.shape

	#accuracy  = model.evaluate({'input1' : trainX, 'input2' : charNPArray, 'output': trainY})                
	#allAccuracy.append(accuracy)
	
	classes = model.predict({'input1': trainX, 'input2': charNPArray})
        accuracy  = evaluateModel(classes, trainY)
	print "Accuracy is : ", str(accuracy)
	allAccuracy.append(accuracy)	

	print "Batch accuracy : ", str(accuracy)

        #model.train_on_batch(newPostArray, newLabelArray)

	del postNPArray, labelNPArray, charNPArray, trainX, trainY
	del gc.garbage[:]

        startIndex = lastIndex
        lastIndex = startIndex + batch_size
	print "start index and last index : ", startIndex, lastIndex
		
print "all epoch finished, saving model"
writeToLogFile(str(allAccuracy))        
