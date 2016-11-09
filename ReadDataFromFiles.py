# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 18:37:29 2016

@author: q409NMNQ
"""

from collections import OrderedDict

class ReadSentences(object):
    def __init__(self, fileName1, fileName2):
        self.fileName1 = fileName1
        self.fileName2 = fileName2
        
        
    def removeZeroLengthWords(self, tokens):
        newTokens = []
        
        for token in tokens:
            if len(token) > 0:
                newTokens.append(token)
        
        return newTokens
    
    def convertIntoOrderedDict(self, tokens, labels):
        
        tokenDict = OrderedDict()
        labelDict = OrderedDict()
        index = 1
        for token, label in zip(tokens, labels):
            #print token, label
            tokenDict[index] = token
            labelDict[index]= float(label)  
            index = index + 1 
            
        return tokenDict, labelDict
        
    def __iter__(self):
        for line1, line2 in zip(open(self.fileName1), open(self.fileName2)):
            tokens = line1.strip().split(' ')
            line21  = line2.replace("[", "").replace("]", "")
            labels = line21.strip().split(',')

            cleanTokens = self.removeZeroLengthWords(tokens)
            cleanLabels = self.removeZeroLengthWords(labels)

            #if len(tokens) != len(labels):
            #   print "token and labels : ", tokens, labels
            #   raise AssertionError()

            if len(cleanTokens) != len(cleanLabels):
               print "clean token and labels : ", cleanTokens, cleanLabels            
               raise AssertionError()

            #if len(cleanTokens > 0) and len(cleanLabels) > 0:
            tokenDict, labelDict = self.convertIntoOrderedDict(cleanTokens, cleanLabels)

            #print tokenDict, labelDict
            yield tokenDict, labelDict   
            
  
def readFiles(filePath1, filePath2):
    iterator = ReadSentences(filePath1, filePath2)

    postDictList = []
    labelDictList = []
    for entry in iterator:
	   #if len(entry[0].values()) < 2:
	   #    print "manual check : ", entry[0], entry[1]
	
	   if len(entry[0].values()) != 0:
            postDictList.append(entry[0])
            labelDictList.append(entry[1])
        
    return postDictList, labelDictList
    

