# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 18:37:29 2016

@author: q409NMNQ
"""

from collections import OrderedDict


def cleanPost(post):

    restrictedCharNS = ['\n', ',', ';', '\\', '(', ')', '[', ']', '{', '}', '<', '>', '"', '?', '‘‘', '’’']
    restrictedCharWS = ['\'', '&', ':', '/','.', '’']

    for eachChar in restrictedCharNS:
        post = post.replace(eachChar, '')

    for eachChar in restrictedCharWS:
        post = post.replace(eachChar, ' ')
    
    #newPost = post.replace('\n','').replace(',',' ').replace('\\','').replace('"','').
    #replace('(', '').replace(')','').replace('\'',' ').replace('/', ' ').replace('.', ' ')

    return post.lower()
    #newX = re.sub('[^a-zA-Z0-9\.]', ' ', x).strip() 

def isNumber(inString):
    return inString.replace('.','',1).replace('-','',1).isdigit()

def replaceNumberWithZeros(post):

    updatedString = ''
    tokens = post.split(' ')
    for token in tokens:
        if isNumber(token):
            updatedString = updatedString + '0' + ' '
        else:
            updatedString = updatedString + token + ' '

    return updatedString.strip() 

class ReadSentences(object):
    def __init__(self, fileName1, fileName2):
        self.fileName1 = fileName1
        self.fileName2 = fileName2
        
    def __iter__(self):
        for line1, line2 in zip(open(self.fileName1), open(self.fileName2)):            
            post = line1.strip()
            keywords = line2.strip()
            cleanedPost = cleanPost(post)
            #postWONumbers = replaceNumberWithZeros(cleanedPost)
            cleanedKeywords = cleanPost(keywords)
            yield cleanedPost, cleanedKeywords
            
  
def readFiles(filePath1, filePath2):
    iterator = ReadSentences(filePath1, filePath2)

    postList = []
    keywordList = []
    for entry in iterator:
        postList.append(entry[0])
        keywordList.append(entry[1])
        
    return postList, keywordList
    

