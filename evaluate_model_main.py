# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 17:24:27 2016

@author: q409NMNQ
"""

from ReadSentencesFromFiles import *
#from ReadDataFromFiles import *
#from PartofSpeechExtraction import *
from NumpyArrayConversion import *
#from GenerateCharEmbeddings import *
from Evaluation import *
from configuration import *
from Utilities import *


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



class LabelPost:
    model = None
    conversionInstance = None

    def __init__(self):        
        self.conversionInstance = NumpyArrayConversion(maxSeqLen, wordEmbeddingSize, wordEmbeddingPath, False, False, charEmbeddingPath, charEmbeddingSize)
        print "conversion instance initialized :"

        self.model = getSingleInputModel (300, 0.0, 0.0, 300, maxSeqLen)
        self.model.load_weights(MODELWEIGHTS)      

        print "Model loaded with weights"

    def generateLabels(self, postDictList, labelDictList):
        postNPArray, labelNPArray = self.conversionInstance.getNumPyArray (postDictList, None, False)
        testX, testY = getTrainingData(postNPArray, labelNPArray)
        print "shape of textx and test Y : ", testX.shape, testY.shape


        y_pred = self.model.predict({'input1' : testX})
        keywordDictList = getKeywords(y_pred, postDictList)

        #f_score, precision, recall = evaluateModel(y_pred, testY)
        #print "eval metrics ", f_score, precision, recall

        return keywordDictList       



def readDatasetDirectory():
    #bookList = ['AI1', 'IP2', 'OS1', 'OS4', 'CN4', 'IP3', 'OS2']
    allBookPaths = []
    allDirs = []
    for (dirpath, dirnames, filenames) in walk(EVALDATASETPATH):
        for dirname in dirnames:
            subDirPath = os.path.join(dirpath, dirname)
            allDirs.append(subDirPath)
            dirFiles = os.listdir(subDirPath)   
            
            for eachFile in dirFiles:
                #if 'CN5' not in os.path.join(subDirPath,eachFile):
                #    continue
                bookPath = {}
                if '.txt' in eachFile:
                    bookPath['dirpath'] = os.path.join(subDirPath, '')
                    bookPath['posts'] = os.path.join(subDirPath, eachFile)                    
            
                if bookPath:
                    allBookPaths.append(bookPath)
                
        
    #print allDirs, len(allDirs)
    #print allBookPaths, len(allBookPaths)
    return allBookPaths


def readDataset():
    
    allBookPaths = readDatasetDirectory()
    print "length of all books dict is : ", len(allBookPaths)

    

    labelPostInstance = LabelPost()
    
    for eachBook in allBookPaths:
        postDictList = readFiles (eachBook['posts'])
        print "Length of post dict list : ", len(postDictList)

        postDictList = cleanPostList(postDictList)
        print "Length of post dict list after cleaning: ", len(postDictList)

        keywordDictList = labelPostInstance.generateLabels(postDictList, [])

        generatedKeywordsPath = eachBook['posts'].replace ('en.txt', '_keywords.txt')

        with open(generatedKeywordsPath, 'w') as f:
            for keywordDict in keywordDictList:
                if len(keywordDict.values()) > 0:
                    f.write(str(keywordDict.values()) + "\n")

        #print "Sample : ", postDictList[0]
        
    

def evaluateUsingSingleFile():

    EVALFILEPATH = "/assignments/IUI/Dataset/OS4/posts.txt"
    LABELFILEPATH = "/assignments/IUI/Dataset/OS4/labels.txt"

    labelPostInstance = LabelPost()    
    postDictList, labelDictList = readFiles (EVALFILEPATH, LABELFILEPATH)
    print "Length of post dict list : ", len(postDictList)

    postDictList, labelDictList = cleanLists(postDictList, labelDictList)
    print "Length of post dict list after cleaning: ", len(postDictList), len(labelDictList)

    keywordDictList = labelPostInstance.generateLabels(postDictList, labelDictList)



#evaluateUsingSingleFile()
readDataset()

