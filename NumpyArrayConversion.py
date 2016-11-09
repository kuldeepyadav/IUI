# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 18:20:19 2016

@author: q409NMNQ
"""
#import fasttext
import pickle
from collections import OrderedDict
import numpy as np

class NumpyArrayConversion:
        
    maxSeqLen = 0
    charEmbeddingSize = 0
    #embeddingSize = 100
    #embeddingPath = "D:\\OnlineCourses\\stack exchange dataset\\stackoverflow\\embeddings_all.pickle"
    def __init__(self, maxLen, embeddingSize, embeddingPath, isDictPickle, isCharEmbed=False, charEmbedPath=None, charEmbedSize=0):
        self.maxSeqLen = maxLen        
        self.embeddingSize = embeddingSize        
        self.isDictPickle = isDictPickle   #flag to check if embedding have to be loaded from pickle        
        if isDictPickle:
            self.embedDict = pickle.load(open(embeddingPath, "rb"))
	    print "loaded dictionary embeddings model", embeddingPath
        else:
            import fasttext
            self.embedDict = fasttext.load_model(embeddingPath)    
	    print "loaded word fastext model ", embeddingPath        
            
        if isCharEmbed:
            self.charEmbedDict = pickle.load(open(charEmbedPath, "rb"))
	    self.charEmbedSize = charEmbedSize
	    print "loaded char embedded dict pickle", charEmbedPath
        
    
    def getWordEmbedding(self, word):        
        try:
            vectors  = self.embedDict[word]
            return True, vectors         
        except Exception as e:
            return  False, [0.0] * self.embeddingSize       # return zeros             
            #print "Keyword not found in the dictionary : " + word
            
    def getCharEmbedding(self, word):
        try:
            vectors  = self.charEmbedDict[word]
            return True, vectors         
        except Exception as e:
            return  False, [0.0] * self.charEmbeddingSize       # return zeros             
            #print "Keyword not found in the dictionary : " + word
    
    def associateCharEmbeddings(self, posts):
    
        newPosts = []
        failCount = 0    
    
        for post in posts:
            newPost = OrderedDict()
            for index, word in post.items():
                result, vectors = self.getCharEmbedding(word)
                newPost[index] = vectors
                if result == False:
                    failCount = failCount + 1
            newPosts.append (newPost)
        
        print "Char Embedding failure count ", failCount, len(newPosts)
        
        return newPosts
        

    def associateWordEmbeddings(self, posts):
    
        newPosts = []
        failCount = 0    
    
        for post in posts:
            newPost = OrderedDict()
            for index, word in post.items():
                result, vectors = self.getWordEmbedding(word)
                newPost[index] = vectors
                if result == False:
                    failCount = failCount + 1
            newPosts.append (newPost)
        
        print "Embedding failure count ", failCount, len(newPosts)
        
        return newPosts
    
    def prepareNumPyArray(self, newPosts, labels):
        postLength = len(newPosts)
    
        npPostArray = np.zeros (shape = (postLength, self.maxSeqLen, self.embeddingSize), dtype = np.float32)
    
        #print "Numpy array shape : ", npArray.shape

        index = 0    
        for post in newPosts:            
            npPostArray[index][:len(post.values())] = post.values()
            index = index + 1
    
        print "Numpy post array shape : ", index, npPostArray.shape
    
        npLabelsArray = np.zeros (shape = (postLength, self.maxSeqLen), dtype = np.float32)
    
        index = 0    
        for label in labels:
            #label_values = label.values() 
            npLabelsArray[index][:len(label.values())] = label.values() 
            #map(lambda x: x + 1, label_values)
            index = index + 1
    
        print "Numpy label array shape : ", index, npLabelsArray.shape
    
        return npPostArray, npLabelsArray    
        
    def getNumPyArray (self, posts, labels):
        newPosts = self.associateWordEmbeddings(posts)
        npPostArray, npLabelsArray = self.prepareNumPyArray(newPosts, labels)
        return npPostArray, npLabelsArray       
        
        
    def prepareCharNumPyArray(self, newPosts):
        postLength = len(newPosts)
    
        npCharArray = np.zeros (shape = (postLength, self.maxSeqLen, self.charEmbeddingSize), dtype = np.float32)
    
        #print "Numpy array shape : ", npArray.shape

        index = 0    
        for post in newPosts:
            npCharArray[index][:len(post.values())] = post.values()
            index = index + 1
    
        print "Numpy char array shape : ", index, npCharArray.shape    

	return npCharArray
        
        
    def getCharNumPyArray(self, posts, charEmbeddingSize):
        self.charEmbeddingSize = charEmbeddingSize
        newPosts = self.associateCharEmbeddings(posts)
        npCharArray = self.prepareCharNumPyArray(newPosts)
        return npCharArray
        
        
        
        
        
        
     

    
    
