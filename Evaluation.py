
import numpy as np

def getRefinedRow(eachRow):
    refinedRow = []
    for eachOutput in eachRow:
        label = np.argmax(eachOutput)
        refinedRow.append(label)

    return refinedRow

def getReverseCategorical(categoricalArray):
    allRefinedRows = []
    for eachRow in categoricalArray:
        refinedRow = getRefinedRow(eachRow)
        allRefinedRows.append(refinedRow)

    return allRefinedRows

def computeRowWisePrecisionRecall(groundTruth, predicted):
    trueNegative = 0
    truePositive = 0
    falseNegative = 0
    falsePositive = 0

    #print ("Ground truth : ", groundTruth)
    #print ("Predicted : ", predicted)

    for i in range(len(groundTruth)):
        for j in range(len(groundTruth[i])):
            if (float(groundTruth[i][j]) == predicted[i][j]) and (float(groundTruth[i][j]) == 1.0):
                truePositive = truePositive + 1
            elif (float(groundTruth[i][j]) == predicted[i][j]) and (float(groundTruth[i][j]) == 0.0):
                trueNegative = trueNegative + 1
            elif (float(groundTruth[i][j]) != predicted[i][j]) and (float(groundTruth[i][j]) == 0.0):
                falsePositive = falsePositive + 1
            else:
                falseNegative = falseNegative + 1

    return truePositive, trueNegative, falsePositive, falseNegative


def evaluateModel(classes, testY):

    allOutputs = classes['output']
    print allOutputs.shape

    allPredictedRows = getReverseCategorical(allOutputs)
    print len(allPredictedRows)
    #print allPredictedRows[0]

    allGroundTruthRows = getReverseCategorical(testY)
    print len(allGroundTruthRows)
    #print allGroundTruthRows[0]

    totalTP, totalTN, totalFP, totalFN = computeRowWisePrecisionRecall(allGroundTruthRows, allPredictedRows)

    if totalTP + totalFP > 0.0:
	precision_score = float(totalTP) / (totalTP + totalFP)
    else:
        precision_score = 0.0

    if totalTP + totalFN > 0.0:
    	recall_score = float(totalTP) / (totalTP + totalFN)
    else:
	recall_score = 0.0

    if precision_score + recall_score > 0.0:
    	f_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    else:
	f_score = 0.0

    return f_score, precision_score, recall_score

