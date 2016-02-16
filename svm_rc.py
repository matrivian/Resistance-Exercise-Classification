#------------------------------------------------------------#
#           RESISTANCE TRAINING CLASSIFICATION               #
# Function: classify testing data into one of the 12 classes #
#           and count how many exercises repeated            #
#                                                            #
# Input: Training Date and Testing Data                      #
# Output: Testing accuracy and repeated times                #
#                                                            #
# Author:  Jun Guo                                           #
# Date:    09/25/2015                                        #
# Version: 1.2                                               #
#          Use SVM to classify the feature data              #
# License: GPL3.0                                            #
#------------------------------------------------------------#

#------------------------------------------------------------#
#       Python Module needed for this project                #
#------------------------------------------------------------#
import os
import glob
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs
import scipy.signal
import scipy.linalg as scialg
from scipy.stats.mstats import moment
from sklearn import svm, metrics

#------------------------------------------------------------#
#                        Resistance Class                    #
#------------------------------------------------------------#

class ResistanceClassification:

    def __init__(self, dataFolder):
        self.folder = dataFolder
        self.totalClsNum = 12
        self.featureNum = 3
        self.filterWindowSize = 91

    def getClassname(self, label):
        """
         Output strings of class name

         Parameters
         ------
         label :       class number

         Returns
         ------
             class names
        """
        if label == 0:
            clsname = 'Bench_Press'
        elif label == 1:
            clsname = 'Shoulder_Press'
        elif label == 2:
            clsname = 'Bicep_Curls'
        elif label == 3:
            clsname = 'Upright_Rows'
        elif label == 4:
            clsname = 'Lateral_Raises'
        elif label == 5:
            clsname = 'Overhead_Triceps_Extensions'
        elif label == 6:
            clsname = 'Kneeling_Triceps_Kickbacks'
        elif label == 7:
            clsname = 'Standing_Bent-over_Rows'
        elif label == 8:
            clsname = 'Kneeling_Bent-over_Rows'
        elif label == 9:
            clsname = 'Squats'
        elif label == 10:
            clsname = 'Forward_Walking_Lunges'
        elif label == 11:
            clsname = 'Calf_Raises'
        return clsname


    def filter(self, rawData, windowSize, windowType = 'hanning'):
        """
         Low Pass Filter Function

         Parameters
         ------
         rawData:             input data
         windowSize:         dimension of smoothing window; should be odd number
         windowType:         window type including 'flat', 'hanning', 'hamming', 'bartlett' 'blackman'

         Returns
         ------
         Low pass filtered data in list
        """

        data = np.squeeze(np.asarray(rawData)) 		# convert matrix to array
        #print data.shape
        processedData = np.r_[2*data[0]-data[windowSize-1::-1],data,2*data[-1]-data[-1:-windowSize:-1]]
        if windowType == 'flat':
            w = np.ones(windowSize, 'd')
        else:
            w = eval('np.' + windowType + '(windowSize)')
        smoothed = np.convolve(w/w.sum(), processedData, mode='same')
        filteredData = smoothed[windowSize:-windowSize+1]
        filteredData = np.array(filteredData).tolist() 	# convert array to list
        return filteredData


    def regroupMat(self, dataMat):
        """
        Regroup matrix of the same label

         Parameters
         ------
         dataMat:            input data matrix

         returns
         ------
             the same dimensional matrix with regrouped label
        """
        regroupedMat = np.zeros((1, 4))
        for label in range(self.totalClsNum):
            for i in range(dataMat.shape[0]):
                if int(dataMat[i,-1]) == label:
                    regroupedMat = np.vstack((regroupedMat, dataMat[i,:]))
        regroupedMat = np.delete(regroupedMat, (0), axis=0)
        #np.savetxt('test.csv', regroupedMat, fmt='%g', delimiter=',')
        return regroupedMat


    def readDataInFolder(self, filename):
        """
        Read in testing data

         Parameters
         ------
         filename:         file name

         Returns
         ------
            matrix has 5 colums, time, X, Y, Z and label
        """
        try:
            dataFile = open(filename, 'r+')
        except IOError:
            print 'Could not open the data file!'
        # Initilization
        dataMat = np.genfromtxt(dataFile, delimiter=',')
        dataFile.close()
        return dataMat


    def largestVar(self, data):
        """
        Find the largest variance among X, Y and Z

         Parameters
         ------
         data :             5 x colNum matrix

         Returns
         ------
               Axis that has the largest variance
        """

        varCoord = [abs(np.var(data[i,:])) for i in range(1,4)]
        return varCoord.index(max(varCoord))


    def detectRange(self, smoothed):
        """
         Detect whether the data is in certain range or not

         Parameters
         ------
         smoothed :              smoothed data 5 x colNum matrix

         Returns
         ------
               indices of range within two sigma of the mean
        """

        mean = np.mean(smoothed)
        var = np.var(smoothed)
        twoSigRange = mean + 1.5*var
        return twoSigRange


    def detectPeaks(self, x, mpd=120, threshold=0):
        """
         Peaks Detection Function

         Parameters
        ----------
         x   :       1D array_like data
         mpd :       positive integer detect peaks that are at least separated by minimum peak distance (in number of data)
         threshold:  peaks - neighbors threshold 

         Returns
        -------
            indeces of the peaks in `x`.
        """

        varIdx = self.largestVar(x)
        smoothed = np.array(x[varIdx+1, :])
        cleanedTime = np.array(x[0, :])

        smoothed = np.atleast_1d(smoothed).astype('float64')
        #print smoothed
        # find indices of all peaks
        dx = smoothed[1:] - smoothed[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(smoothed))[0]
        if indnan.size:
            smoothed[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == smoothed.size-1:
            ind = ind[:-1]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([smoothed[ind]-smoothed[ind-1], smoothed[ind]-smoothed[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(smoothed[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (smoothed[ind[i]] > smoothed[ind])
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])
        peakIdx = [item for item in ind if smoothed[item] > self.detectRange(smoothed)]
        peakList = smoothed[peakIdx]
        timeList = cleanedTime[peakIdx]
        return (peakList, timeList, peakIdx)


    def timeShift(self, tmMatrix):
        """
        Shift time interval to [0: end of interval]

         Parameters
         ------
         tmMatrix :         time list

         Returns
         ------
               shifited time array starting from 0
        """

        tm = list(np.array(tmMatrix).reshape(-1,))
        cycle = 60
        end = 59.9
        if (tm[-1]-tm[0]) > 0:
            shift = tm[0]
            for i in range(len(tm)):
                tm[i] -= shift
        elif (tm[-1]-tm[0]) < 0:
            shift = tm[0]
            for i in range(len(tm)):
                if tm[i] >= shift and tm[i] <= end:
                    tm[i] -= shift
                elif tm[i] < shift:
                    tm[i] += cycle
                    tm[i] -= shift
        return tm


    def extractFeature(self, dataMat):
        """
        Feature Extraction using moments

         Parameters
         ------
         dataMat:        input matrix which is rows # x 5 columns

         Return
         ------
                48 x 3 matrix containing four moments
                48: 12 classes x 4 moments
                3:  X, Y and Z
        """
        featureMat = np.zeros((1,4))
        for label in range(self.totalClsNum):
            startInd = np.where(dataMat[:,-1] == label)[-1][0]
            endInd = np.where(dataMat[:,-1] == label)[-1][-1]
            m1 = np.mean(dataMat[startInd:endInd, 1:4], axis=0)
            m1 = np.append(m1, label)
            m2 = np.var(dataMat[startInd:endInd, 1:4], axis=0)
            m2 = np.power(m2, 1.0/2)
            m2 = np.append(m2, label)
            m3 = moment(dataMat[startInd:endInd, 1:4], 3, axis=0)
            m3 = np.power(abs(m3), 1.0/3)*np.sign(m3)
            m3 = np.append(m3, label)
            m4 = moment(dataMat[startInd:endInd, 1:4], 4, axis=0)
            m4 = np.power(abs(m4), 1.0/4)
            m4 = np.append(m4, label)
            featureMat = np.vstack((featureMat, m1))
            featureMat = np.vstack((featureMat, m2))
            featureMat = np.vstack((featureMat, m3))
            featureMat = np.vstack((featureMat, m4))
        featureMat = np.delete(featureMat, (0), axis=0)
        return featureMat


    def training(self):
        """
        Read in training file, extract feature and regroup the feature matrix

        Parameter:
        ------

        Return:
        ------
            svm classifier
        """
        trainingDataFeature = np.zeros((1,4))
        for dataFilename in glob.glob(self.trainingDataFolder + '*.csv'):
            rawData = self.readDataInFolder(dataFilename)
            filteredData = np.zeros((rawData.shape[0], rawData.shape[1]))
            filteredData[:,0] = rawData[:,0]; filteredData[:,4] = rawData[:,4]
            for i in range(1, 4):
                filteredData[:,i] = self.filter(rawData[:,i], self.filterWindowSize)
            tmpFeature = self.extractFeature(filteredData)
            trainingDataFeature = np.vstack((trainingDataFeature, tmpFeature))
        traingDataFeature = np.delete(trainingDataFeature, (0), axis=0)
        trainingFeatureMat = self.regroupMat(traingDataFeature)

        # SVM training
        C = 1.0 # SVM regularization parameter
        clf = svm.SVC(kernel='rbf', gamma=300, C=C)
        clf.fit(trainingFeatureMat[:,0:self.featureNum], trainingFeatureMat[:,self.featureNum])
        return clf


    def getResult(self, clfResult):
        """
        Reduce the classification result to a vector containing 0-11

        Parameter:
        ------
            clfResult:           SVM classification result

        Return:
        -----
            result:              reduced dimension vector 1 x 12
        """
        start = 0; end = 4; interval = 4
        result = []
        for i in range(12):
            count = np.bincount(clfResult[start:end].astype(int))
            if max(count) == 1:
                result.append(clfResult[start].astype(int))
            else:
                ind = np.argmax(count)
                result.append(ind)
            start += interval
            end += interval
        return result


    def svmClf(self):
        """
        SVM classifier main function

        Parameter:
        ------

        Return:
        ------
            Print the result of the classification
        """
        finalResult, finalExpected = [], []
        for testFileName in glob.glob(self.folder + '*.csv'):
            rawTestData = self.readDataInFolder(testFileName)
            filteredTestData = np.zeros((rawTestData.shape[0], rawTestData.shape[1]))
            filteredTestData[:,0] = rawTestData[:,0]; filteredTestData[:,4] = rawTestData[:,4]
            for i in range(1, 4):
                filteredTestData[:,i] = self.filter(rawTestData[:,i], self.filterWindowSize)
            testDataFeature = self.extractFeature(filteredTestData)
            trainingDataFeature = np.zeros((1,4))
            for restFileNames in glob.glob(self.folder + '*.csv'):
                if restFileNames != testFileName:
                    rawTrainData = self.readDataInFolder(restFileNames)
                    filteredTrainData = np.zeros((rawTrainData.shape[0], rawTrainData.shape[1]))
                    filteredTrainData[:,0] = rawTrainData[:,0]; filteredTrainData[:,4] = rawTrainData[:,4]
                    for i in range(1, 4):
                        filteredTrainData[:,i] = self.filter(rawTrainData[:,i], self.filterWindowSize)
                    tmpTrainFeature = self.extractFeature(filteredTrainData)
                    trainingDataFeature = np.vstack((trainingDataFeature, tmpTrainFeature))
                traingDataFeature = np.delete(trainingDataFeature, (0), axis=0)
                trainingFeatureMat = self.regroupMat(traingDataFeature)
            # SVM training
            C = 1.0 # SVM regularization parameter
            clfr = svm.SVC(kernel='rbf', gamma=300, C=C)
            clfr.fit(trainingFeatureMat[:,0:self.featureNum], trainingFeatureMat[:,self.featureNum])
            # SVM testing
            clfResult = clfr.predict(testDataFeature[:,0:self.featureNum])
            finalResult += self.getResult(clfResult)
            finalExpected += range(12)
            #print finalResult, finalExpected
        print("Classification report:\n%s") %(metrics.classification_report(finalExpected, finalResult))
        print("Confusion Matrix:\n%s") %(metrics.confusion_matrix(finalExpected, finalResult))


#------------------------------------------------------------#
#                 Run it from here                           #
#------------------------------------------------------------#
if __name__ == "__main__":
    import sys
    exercise = ResistanceClassification(sys.argv[1])
    exercise.svmClf()
