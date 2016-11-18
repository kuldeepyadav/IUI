# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:30:20 2016

@author: q409NMNQ
"""

from keras.utils.np_utils import to_categorical
import numpy as np
import random

def getTrainingData(postNPArray, labelNPArray):
    
    cLabelNPArray = np.zeros (shape=(labelNPArray.shape[0], labelNPArray.shape[1], 2), dtype = np.float32)
    for i in range(labelNPArray.shape[0]):
        cLabelNPArray[i] = to_categorical(labelNPArray[i], 2)    

    # Data preprocessing

    totalX = postNPArray
    totalY = cLabelNPArray
   
    #print "Size of total vectors : ", totalX.shape, totalY.shape
    #print "Size of testing vectors : ", testX.shape, testY.shape
    #print ("testy.shape", testY[0])
    return totalX, totalY  

def convertDictToPost(postDict):

    post = ''
    for index, word in postDict.items():
        post = post + str(word) + ' '

    return post.strip()

def findBigggestSequence(labels):
    
    maxLen = -1
    for label in labels:
        if len(label.values()) > maxLen:
            maxLen = len(label.values())
    
    print "Maximum length of subsequence is : ", maxLen
    
    return maxLen

def cleanLists(postDictList, labelDictList, filterAllZeros = False):

    newPostDictList = []
    newLabelDictList = []

    numShortPosts = 0
    numLongPosts = 0    

    for postDict, labelDict in zip(postDictList, labelDictList):
	       
        assert len(postDict.values()) == len(labelDict.values())

	    #print len(postDict.values()), len(labelDict.values())
        if len(postDict.values()) < 4:
           numShortPosts = numShortPosts + 1
           continue

        if len(postDict.values()) >= 100:
           numLongPosts = numLongPosts + 1
           print "Long list : ", convertDictToPost(postDict)
           continue

        if sum(labelDict.values()) == 0 and filterAllZeros:
        	continue

	    #wordLengths = []
	    #for word in postDict.values():
	    #    wordLengths.append(len(word))

	    #avgLength = sum(wordLengths)/len(wordLengths)
	
	    #if avgLength > 8:
	    #   print "Skipping this one: ", postDict
	    #   continue

        newPostDictList.append(postDict)
        newLabelDictList.append(labelDict)

    print "Num of short and long posts are : ", numShortPosts, numLongPosts
    return newPostDictList, newLabelDictList

def shuffleLists (postDictList, labelDictList):

    combinedList = list(zip(postDictList, labelDictList))

    random.shuffle(combinedList)

    postDictList, labelDictList = zip(*combinedList)

    return postDictList, labelDictList



def  filterLists(postDictList, labelDictList):
    newPostDictList = []
    newLabelDictList = []

    for postDict, labelDict in zip(postDictList, labelDictList):
        assert len(postDict.values()) == len(labelDict.values())

        #print len(postDict.values()), len(labelDict.values())

  	if len(postDict.values()) >= 200:
           continue     

        newPostDictList.append(postDict)
        newLabelDictList.append(labelDict)

    return newPostDictList, newLabelDictList

def findLabelDistribution(labelDictList):

    labelExist = 0
    for labelDict in labelDictList:
        total = sum(labelDict.values())
        if total > 0:
           labelExist = labelExist + 1

    print "Label exists in : ", len(labelDictList), labelExist

def findSeqLenDistribution(labels):
    
    seqLengths = []
    moreThan1K = 0
    for label in labels:
        length = len(label.values())
	if length > 100:
	   moreThan1K = moreThan1K + 1
	elif length == 0:
	   print "Zero length : ", label
	seqLengths.append(length)

    #print "Maximum length of subsequence is : ", maxLen    
    avgLength = sum(seqLengths)/len(seqLengths)

    print  min(seqLengths), max(seqLengths), avgLength, moreThan1K


def getMaxWordLength(posts):
    maxWordLength = 0
    maxLenWord=""
    maxPostDict = None
    for postDict in posts:
        for word in postDict.values():
            if (len(word) > maxWordLength):
                maxWordLength, maxLenWord, maxPostDict  = len(word), word, postDict

    print "Max word length : ", maxWordLength, len(maxLenWord)
    if maxWordLength%2 == 1:
        maxWordLength = maxWordLength + 1
    print "Max word length : ", maxWordLength
    return maxWordLength, maxLenWord, maxPostDict


def getUniqueWords(posts):
    uniqueWords = []

    for postDict in posts:
        for word in postDict.values():
            if word not in uniqueWords:
                uniqueWords.append(word)

    return uniqueWords

def getBigSequences (postDictList, labelDictList):    

    newPosts = []
    newLabels = []

    for postDict, labelDict in zip(postDictList, labelDictList):
	length = len(postDict.values())
	if length >= 50:
	   newPosts.append(postDict)
	   newLabels.append(labelDict)

    print "Length of big seequences : ", len(newPosts), len(newLabels)	
    return newPosts, newLabels

def dumpToFile(posts, filePath = '', fileName = 'posts.txt'):

    with open(filePath + fileName, "w") as f:
        for post in posts:
            f.write(str(post) + "\n")

def dumpLabelsToFile(labelDictList, filePath = '', fileName = 'labels.txt'):

    with open(filePath + fileName, "w") as f:
        for labelDict in labelDictList:
            f.write(str(labelDict.values()).replace('[', '').replace(']', '') + "\n")

def dumpPostTaggedWithLabels(postDictList, labelDictList, filePath = '', fileName = 'printedOutput.txt'):

    assert len(postDictList) == len(labelDictList)

    with open(filePath + fileName, "w") as f:
        for postDict, labelDict in zip(postDictList, labelDictList):
            fullPost = ''
            for index, eachToken in postDict.items():
                if labelDict[index] == 1:
                    fullPost = fullPost + '**' + eachToken + '**' + ' '
                else:
                    fullPost = fullPost + eachToken + ' '
            f.write (str(fullPost.strip()) + '\n')


