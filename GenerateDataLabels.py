# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 17:24:27 2016

@author: q409NMNQ
"""

#import sys
#sys.path.insert(0, r'/home/kuldeep/keras')

import nltk
from nltk import metrics, stem, tokenize
from nltk.metrics.distance import edit_distance
from nltk.corpus import stopwords


from ReadPreProcessedDataFromFiles import *
#from PartofSpeechExtraction import *
#from NumpyArrayConversion import *
#from GenerateCharEmbeddings import *
from configuration import *

from Utilities import *
from os import walk
import os
import pickle
import time


WordEmbedPATH = ""

#filePath1 = DATABASEPATH + "murphy_sentence-words.csv"
#filePath2 = DATABASEPATH + "murphy_sentence-target.csv"

stemmer = None
tagDict = None
stopWordList = None

def initializeGlobalVariables():
    global stemmer
    stemmer = stem.PorterStemmer()    
    
    global stopWordList
    stopWordList = set(stopwords.words('english')) 

def normalize(s):    
    words = tokenize.wordpunct_tokenize(s.lower().strip())
    return ' '.join([stemmer.stem(w) for w in words])

def exact_match(s1, s2):
    s1_new = s1.decode('utf8')
    s2_new = s2.decode('utf8')

    if s1_new.strip() == s2_new.strip():
       return True
    elif normalize(s1_new) == normalize(s2_new):
       return True
    else:
       return False

 
def fuzzy_match(s1, s2, max_dist=1):
    s1_new = s1.decode('utf8')
    s2_new = s2.decode('utf8')

    dist =  edit_distance(normalize(s1_new), normalize(s2_new))
    maxLength = max(len(s1_new), len(s2_new))
    minLength = min(len(s1_new), len(s2_new))
    
    if maxLength <= 3:
        if dist == 0:
            return True
        else:
            return False
    
    if (dist <= max_dist) and (maxLength-minLength) <= 2:
        return True
    else:
        return False


def dumpToFile(posts, filePath = '', fileName = 'posts.txt'):

    with open(filePath + fileName, "w") as f:
        for post in posts:
            f.write(str(post) + "\n")

def dumpLabelsToFile(labelDictList, filePath = '', fileName = 'labels.txt'):

    with open(filePath + fileName, "w") as f:
        for labelDict in labelDictList:
            f.write(str(labelDict.values()).replace('[', '').replace(']', '') + "\n")

def dumpPostTaggedWithLabels(postDictList, labelDictList, filePath = '', fileName = 'postsWithLabels.txt'):

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


def readDatasetDirectory():
    bookList = ['OS2', 'OS4']
    allBookPaths = []
    allDirs = []
    
    print "Reading dataset : ", DATABASEPATH

    for (dirpath, dirnames, filenames) in walk(DATABASEPATH):
        for dirname in dirnames:
            subDirPath = os.path.join(dirpath, dirname)
            allDirs.append(subDirPath)
            dirFiles = os.listdir(subDirPath)	
            bookPath = {}
            for eachFile in dirFiles:
		print "Reading : ", eachFile
		if 'OS2' not in os.path.join(subDirPath, eachFile):
		    print "Found OS"
		    continue
                if 'xx2' in eachFile:
                    bookPath['dirpath'] = os.path.join(subDirPath, '')
                    bookPath['posts'] = os.path.join(subDirPath, eachFile)                    
                elif 'zz2' in eachFile:
		    bookPath['keywords'] = os.path.join(subDirPath, eachFile)
            
            if bookPath:
                allBookPaths.append(bookPath)
                
	    
    #print allDirs, len(allDirs)
    #print allBookPaths, len(allBookPaths)
    return allBookPaths

def readDataset():
    
    allBookPaths = readDatasetDirectory()
    print "length of all books dict is : ", len(allBookPaths)
    
    for eachBook in allBookPaths:
        print eachBook
        postList, keywordList = readFiles(eachBook['posts'], eachBook['keywords'])
        print "length of : ", len(postList), len(keywordList)

        dumpToFile(postList, eachBook['dirpath'], "posts.txt")
        dumpToFile(keywordList, eachBook['dirpath'], "keywords.txt")

        postDictList = []
        labelDictList= []
	uniqueKeywordList = list(set(keywordList))
	print "number of unique keywords are : ", len(uniqueKeywordList)
	dumpToFile(uniqueKeywordList, eachBook['dirpath'], "uniqueKeywords.txt")

	i = 0
	starttime = time.time()
        for eachPost, eachKeyword in zip(postList, keywordList):
            postDict, labelDict = findLabelDictForPost(eachPost, eachKeyword, uniqueKeywordList)
            postDictList.append(postDict)
            labelDictList.append(labelDict)
	    if i % 1000 == 0:
	       diffTime = time.time() - starttime
	       starttime = time.time()
	       print "Finished : ", i, diffTime
	       #break
	    i = i + 1

        dumpPostTaggedWithLabels(postDictList, labelDictList, eachBook['dirpath'], 'printedOutput.txt')
        dumpLabelsToFile(labelDictList, eachBook['dirpath'], 'labels.txt')

        print "length of updated dict : ", len(postDictList), len(labelDictList)


def findLabelDictForPost(postTitle, keyword, keywordList):
    
    initialTokens = postTitle.split(' ')
    tags = keyword.split(' ')
    
    postTokens = []
    for postToken in initialTokens:
        if (len(postToken.strip()) > 0):
            postTokens.append(postToken)
            
    postDict = OrderedDict()
    labelDict = OrderedDict()
    
    i = 1
    for token in postTokens:
        postDict[i] = token
        labelDict[i] = 0
        i = i + 1
    
    for key,token in postDict.items():
        if token in stopWordList:
            continue
        
        for tag in tags:
            if fuzzy_match(tag, token):                   
                labelDict.update({key: 1}) 
                 
        #print "post dict 1 : ", postDict, labelDict

        for tag in tags:
            dashCount = tag.count('-')                            
        
            if dashCount == 0:
                continue
        
            tagTokens = tag.split('-')
        
            count = 0
            newDict = {}
            for tagToken in tagTokens:
                for key, token in postDict.items():
                    if fuzzy_match(tagToken, token):                    
                        count = count + 1
                        newDict.update({key: 1})
                    
                    #print tagToken, token, count
        
            if (dashCount+1) == count:
                #print newDict 
                labelDict.update(newDict) 
        
        
    #print "Post dict 2 is : ", postDict, labelDict 

    #f = open("matchinglogs.txt", "a")
    #f.write ("post : " +  convertDictToPost(postDict) + "\n")

    for keyword in keywordList:
        
	#f.write ("matching keyword : " + keyword + "\n")
	keywordTokens = keyword.split (' ')

	newKeywordTokens = []
	for eachToken in keywordTokens:
	    if len(eachToken.strip()) > 0:
	       newKeywordTokens.append(eachToken)

	keywordTokens= newKeywordTokens
	
	total_matches = 0
        matched_words = []
	indexes = []

	for eachToken in keywordTokens:
	    for index, value in postDict.items():
		if exact_match (eachToken, value):
	           total_matches = total_matches + 1
		   indexes.append(index)
		   matched_words.append(eachToken)

	#uniquedMatchedWords = List(set(matched_words))
	commonWords = list(set(matched_words) & set(keywordTokens))
	#f.write ("post : " +  convertDictToPost(postDict) + "\n")
	#f.write ("matched words : " + str(matched_words) + "   ") 
	#f.write ("keyword tokens : " + str(keywordTokens) + "\n")

	if len(commonWords) == len(keywordTokens):
	   #print "post : ", convertDictToPost(postDict), len(commonWords)
	   #print  "matched : ", keywordTokens, indexes, matched_words
	   for index in indexes:
	       labelDict[index] = 1
	
    #f.close()	
    return postDict, labelDict   


initializeGlobalVariables()
readDataset()




