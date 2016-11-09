# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 00:03:31 2016

@author: q409NMNQ
"""

from spacy.en import English
from collections import OrderedDict
import numpy as np


def constructSentence(postDict):
    
    sentence = ""
    
    for eachWord in postDict.values():
        sentence = sentence + eachWord + " "
        
    strippedSentence = sentence.strip()
    
    return strippedSentence
    
def convertListInUnicode(tokens):
    
    newTokens = []
    for token in tokens:
        newTokens.append(unicode(token, "utf-8"))
        
    return newTokens
    
def findAllTags(poSpeechDictList):
    
    taggerTags = []
    dependencyTags = []
    posTags = []

    for poSpeechDict in poSpeechDictList:
        for eachValue in poSpeechDict.values():
            posTags.append(eachValue[0])
            taggerTags.append(eachValue[3])
            dependencyTags.append(eachValue[5])
        

    print len(posTags), len(taggerTags), len(dependencyTags)
    
def getSentenceFeatures(postDictList):

    nlp = English()   
    
    poSpeechDictList = []
    
    for postDict in postDictList:
       #sentence = constructSentence(postDict)
       #unicodeSentence = unicode(sentence, "utf-8")
       unicodeTokens = convertListInUnicode(postDict.values())
       
       doc = nlp.tokenizer.tokens_from_list(unicodeTokens)
       nlp.parser(doc)
       nlp.tagger(doc)
       
       #formattedTokens = nlp(unicodeSentence)
       poSpeechDict = OrderedDict()
       
       index = 1
       for token in doc:
           poSpeechDict[index] = [token.pos_, token.pos, token.tag, token.tag_, token.dep, token.dep_]
           index = index + 1
           
       poSpeechDictList.append(poSpeechDict)
       
    return poSpeechDictList

def getNPArrayOfSentenceFeatures(postDictList, maxSeqLen, rangeMin=-1.0, rangeMax = 1.0):

    posSpeechDictList = getNormalizedSentenceFeatures(postDictList, rangeMin=-1.0, rangeMax = 1.0)

    numFeatures = 3

    posNPArray  = np.zeros(shape=(len(postDictList), maxSeqLen, numFeatures), dtype = np.float32)    

    index = 0
    for eachDict in posSpeechDictList:
        posNPArray[index, :len(eachDict.values())] = eachDict.values()
	index = index + 1

    return posNPArray  
   

def getNormalizedSentenceFeatures(postDictList, rangeMin=-1.0, rangeMax = 1.0):   
    
    poSpeechDictList = getSentenceFeatures(postDictList)

    tagArray = []
    parserArray = []
    dependencyArray = []

    for poSpeechDict in poSpeechDictList:
        for eachValue in poSpeechDict.values():
            tagArray.append(int(eachValue[1]))
            parserArray.append(int(eachValue[2]))
            dependencyArray.append(int(eachValue[4]))
            
            
    tagMin = min(tagArray)
    tagMax = max(tagArray)
    parserMin = min(parserArray)
    parserMax = max(parserArray)
    depMin = min(dependencyArray)
    depMax = max(dependencyArray)   
    
    print "Min max: ", tagMin, tagMax, parserMin, parserMax, depMin, depMax
    
    tagA = float(rangeMax-rangeMin)/float(tagMax-tagMin)
    tagB =  rangeMin - (tagA * tagMin)

    parserA = float(rangeMax-rangeMin)/float(parserMax-tagMin)
    parserB =  rangeMin - (parserA * parserMin)
    
    depA = float(rangeMax-rangeMin)/float(depMax-depMin)
    depB =  rangeMin - (depA * depMin)
    
    newPoSpeechDict=[]
    
    for poSpeechDict in poSpeechDictList:        
        posDict = OrderedDict()
        for key,eachValue in poSpeechDict.items():
            newTagValue = tagA * int(eachValue[1]) + tagB            
            newParserValue= parserA * int(eachValue[2]) + parserB
            newDepValue = depA * int(int(eachValue[4])) + depB
            posDict[key] = [newTagValue, newParserValue, newDepValue]
            
        newPoSpeechDict.append(posDict)
        
    return newPoSpeechDict
            


    
    
    
            
    
    
                       
       

