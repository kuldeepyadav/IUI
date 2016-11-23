# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 18:37:29 2016

@author: q409NMNQ
"""

from collections import OrderedDict

class ReadSentences(object):
    def __init__(self, fileName1):
        self.fileName1 = fileName1
        
        
        
    def removeZeroLengthWords(self, tokens):
        newTokens = []
        
        for token in tokens:
            if len(token) > 0:
                newTokens.append(token)
        
        return newTokens
    
    def convertIntoOrderedDict(self, tokens):
        
        tokenDict = OrderedDict()
        
        index = 1
        for token in tokens:
            #print token, label
            tokenDict[index] = token
            index = index + 1 
            
        return tokenDict
        
    def __iter__(self):
        for line1 in open(self.fileName1):
            tokens = line1.strip().split(' ')
            cleanTokens = self.removeZeroLengthWords(tokens)
  	    tokenDict = self.convertIntoOrderedDict(cleanTokens)

            #print tokenDict, labelDict
            yield tokenDict  
            
  
def readFiles(filePath1):
    iterator = ReadSentences(filePath1)

    postDictList = []
    
    for entry in iterator:
	#if len(entry[0].values()) < 2:
	#    print "manual check : ", entry[0], entry[1]
	postDictList.append(entry)
           
        
    return postDictList
    

