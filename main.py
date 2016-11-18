# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 17:24:27 2016

@author: q409NMNQ
"""

#import sys
#sys.path.insert(0, r'/home/kuldeep/keras')

from ReadDataFromFiles import *
#from PartofSpeechExtraction import *
from NumpyArrayConversion import *
from GenerateCharEmbeddings import *
from configuration import *

from Utilities import *
from os import walk
import os
import pickle


WordEmbedPATH = ""

#filePath1 = DATABASEPATH + "murphy_sentence-words.csv"
#filePath2 = DATABASEPATH + "murphy_sentence-target.csv"

def readDatasetDirectory():
    #bookList = ['AI1', 'IP2', 'OS1', 'OS4', 'CN4', 'IP3', 'OS2']
    allBookPaths = []
    allDirs = []
    for (dirpath, dirnames, filenames) in walk(DATABASEPATH):
        for dirname in dirnames:
            subDirPath = os.path.join(dirpath, dirname)
            allDirs.append(subDirPath)
            dirFiles = os.listdir(subDirPath)   
            bookPath = {}
	    
            for eachFile in dirFiles:
		#if 'CN5' not in os.path.join(subDirPath,eachFile):
		#    continue
                if 'posts' in eachFile:
                    bookPath['posts'] = os.path.join(subDirPath, eachFile)                    
                elif 'labels' in eachFile:
                    bookPath['labels'] = os.path.join(subDirPath, eachFile)
            
            if bookPath:
                allBookPaths.append(bookPath)
                
        
    #print allDirs, len(allDirs)
    #print allBookPaths, len(allBookPaths)
    return allBookPaths


def readDataset():
    
    allBookPaths = readDatasetDirectory()
    print "length of all books dict is : ", len(allBookPaths)
    
    postDictList = []
    labelDictList= []
    for eachBook in allBookPaths:
   	    posts, labels = readFiles (eachBook['posts'], eachBook['labels'])
   	    postDictList.extend(posts)
   	    labelDictList.extend(labels)
   	    print "length of postdictlist and labelDictList: ", len(postDictList), len(labelDictList), len(posts), len(labels)
   	    #print postDictList[0], labelDictList[0]

    newPostDict, newLabelDict = cleanLists(postDictList, labelDictList)
    picklePath = DATABASEPATH + "postLabelDict.pickle"
    postlabelDict = {"posts": newPostDict, "labels": newLabelDict}
    pickle.dump(postlabelDict, open(picklePath, "wb" ))
    print "Dumped postlabeldict to : ", picklePath

    return newPostDict, newLabelDict


def loadDataset(): 

    picklePath = DATABASEPATH + "postLabelDict.pickle"
    postLabelDict = pickle.load(open(picklePath, "rb" ))
    return postLabelDict["posts"], postLabelDict["labels"]

def generateCharEmbeddings(postDictList):
    
    charEmbedPath = PATH + "charEmbedPath.pickle"
    GenerateCharEmbeddings(postDictList, charEmbedPath)


#readDatasetDirectory()
#readDataset()
#postDictList, labelDictList = loadDataset()
#print "Length of data : ", len(postDictList), len(labelDictList)
#cleanedPosts, cleanedLabels = cleanLists(postDictList, labelDictList)

postDictList, labelDictList = readDataset()
print "length of postdictlist and labelDictList: ", len(postDictList), len(labelDictList)

#picklePath = DATABASEPATH + "postLabelDict.pickle"
#postlabelDict = {"posts": postDictList, "labels": labelDictList}
#pickle.dump(postlabelDict, open(picklePath, "wb" ))
#print "Dumped postlabeldict to : ", picklePath

findLabelDistribution(labelDictList)

#maxSeqLen = findBigggestSequence(labelDictList)
#print "Max seq len is : ", maxSeqLen

#findSeqLenDistribution(labelDictList)

#uniqueWords = getUniqueWords(postDictList)
#uniqueWords.sort(key = len)
#wordIterator = reversed(uniqueWords)

#with open(PATH + "uniqueWords.txt", "w") as f:
#     for eachWord in wordIterator:
#	f.write(eachWord +  " " + str(len(eachWord)) + "\n")


#generateCharEmbeddings(postDictList)


#lengthPosts, lengthLabels = getBigSequences (cleanedPosts, cleanedLabels)

#f = open(PATH + "lengthPosts.txt", "w")
#f.write(str(lengthPosts))
#f.close()

#f = open(PATH + "lengthLabels.txt", "w")
#f.write(str(lengthLabels))
#f.close()

#length, word, maxPostDict  = getMaxWordLength(postDictList)

#print "length is : ", length, word, maxPostDict


#poSpeechDictList = getNormalizedSentenceFeatures(postDictList)

#picklePath = PATH + "poSpeechDict.pickle"
#pickle.dump(poSpeechDictList, open(picklePath, "wb" ))

#print poSpeechDictList[0]
#print postDictList[0]
#print labelDictList[0]


#conversionInstance = NumpyArrayConversion(maxSeqLen, wordEmbeddingSize, wordEmbeddingPath, False)
#postNPArray, labelNPArray = conversionInstance.getNumPyArray (postDictList, labelDictList)
#charNPArray = conversionInstance.getCharNumPyArray(postDictList, charEmbeddingSize)


#print "Shape of postNP and label NP array: ", postNPArray.shape, labelNPArray.shape




        
