# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:15:06 2016

@author: q409NMNQ
"""

import numpy as np
import pickle

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

class GenerateCharEmbeddings(object):
    
    def __init__(self, posts, dumpingPath):
        self.posts = posts
        self.maxWordLength = self.getMaxWordLength()        
        self.charVocabLength, self.charVocab = self.getChracterVocabLength()
        self.chartoIndexDict = self.getCharConversionDict()
        self.uniqueWords = self.getUniqueWords()
        self.charEmbeddingSize = 50
        self.dumpingPath = dumpingPath
        charEmbedDict = self.createCharEmbeddings()  
        self.saveEmbeddings(charEmbedDict)
        
        
     
    def getMaxWordLength(self):
         maxWordLength = 0
         for postDict in self.posts:
             for word in postDict.values():
                 if (len(word) > maxWordLength):
                     maxWordLength = len(word)
         
	 print "Max word length : ", maxWordLength
	 if maxWordLength%2 == 1:
	    maxWordLength = maxWordLength + 1
	 print "Max word length : ", maxWordLength
         return maxWordLength

    def getChracterVocabLength(self):    
        allChars =''
        for postDict in self.posts:
            for word in postDict.values():
                for eachChar in word:
                    allChars += eachChar
                
        unique=set(allChars)                #-set('@!#.') add the characters that you wanna neglect in the second set

	print "Number of unique characters: ", len(unique)+1, unique
    
        return len(unique)+1, unique

    def getCharConversionDict(self):
        index = 1
    
        chartoIndexDict = {}
        for char in self.charVocab:
            chartoIndexDict[char] = index 
            index = index + 1
        
        return chartoIndexDict

    def wordtoIndex(self, word):
        indexSeq = []
        index = 0
    
        #print "words is : ", word
        for eachChar in word:
            indexSeq.append(self.chartoIndexDict[eachChar])        
            index = index + 1
     
        for i in range(index, self.maxWordLength):
            indexSeq.append(0)       
        
        #print "length of indexseq ", index, max_charlen, len(indexSeq)
        
        return indexSeq

    def getUniqueWords(self):
        uniqueWords = []

        for postDict in self.posts:    
            for word in postDict.values():
                if word not in uniqueWords:
                    uniqueWords.append(word)

	print "Length of unique words : ", len(uniqueWords)
                
        return uniqueWords
        
    def compileModel(self, maxLen):
        nb_filters = 10
	output_dim = self.maxWordLength*2 + (self.maxWordLength/2)
	print "output_dim is : ", output_dim

        model_cnn = Sequential()
        model_cnn.add(Embedding(self.charVocabLength, self.charEmbeddingSize, input_length=maxLen*self.maxWordLength))
        model_cnn.add(Reshape((maxLen, self.maxWordLength, self.charEmbeddingSize)))
        model_cnn.add(Permute((3,1,2)))
        model_cnn.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
        model_cnn.add(Permute((2,1,3)))
        model_cnn.add(MaxPooling2D((2, 2)))
        model_cnn.add(Reshape((maxLen, output_dim)))
        model_cnn.compile('rmsprop', 'mse')
        
        print "Model_CNN is compiled successfully : "
        
        return model_cnn, output_dim
        
    def createCharEmbeddings(self):
        total_size = len(self.uniqueWords) + 1
        #batch_size, max seq len, index of characters 
        maxLen = 64
        rem = total_size % maxLen
        total_size = total_size + (maxLen - rem)

	print "Rem and total size : ", rem, total_size
        
        npCharArray = np.zeros (shape = (total_size, self.maxWordLength), dtype = np.float32)
        index = 0
        for uniqueWord in self.uniqueWords:
            npCharArray[index] = self.wordtoIndex(uniqueWord)   

	batch_size = int(total_size/maxLen)
	print "Batch size is : ", batch_size, npCharArray.shape

        newCharArray = npCharArray.reshape(batch_size, maxLen, self.maxWordLength)
        newCharArray2 = newCharArray.reshape(batch_size, maxLen*self.maxWordLength)
        
        model, output_dim  = self.compileModel(maxLen)
        
        output_array = model.predict(newCharArray2)
        output_array = output_array.reshape(batch_size*64, output_dim)
        print output_array.shape
        
        index=0
        charEmbedDict= {}
        for uniqueWord in self.uniqueWords:
            charEmbedDict[uniqueWord] = output_array[index]
            index = index + 1

        charEmbedDict[None] = output_array[index]

        print "len of chracterEmbedDict ", len(charEmbedDict)
        
        return charEmbedDict
        
    def saveEmbeddings(self, charEmbedDict):
        pickle.dump(charEmbedDict, open(self.dumpingPath, "wb" ))
        print "Dumped char embedding to : ", self.dumpingPath
       
       
     
