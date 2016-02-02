#------------------------------------------------------------# 
# RESISTANCE TRAINING CLASSIFICATION using cosine similarity #
# Function: classify testing data into one of the 12 classes #
#           and count how many exercises repeated            #
#                                                            #
# Input: Training Date and Testing Data                      #
# Output: Testing accuracy and repeated times                #
#                                                            #
# Author:  Jun Guo                                           #
# Date:    11/13/2014                                        #
# Version: Beta8.0                                           #
#          Add more data to training sets                    #
# License: GPL3.0                                            #
#------------------------------------------------------------# 

#------------------------------------------------------------#
#       Python Module needed for this project                #
#------------------------------------------------------------#
import os
import glob
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs
import scipy.signal
import scipy.linalg as scialg
from scipy.stats.mstats import moment


#------------------------------------------------------------#
#                        Functions                           #
#------------------------------------------------------------#

class ResistanceClassification:

    def __init__(self, trainingData, testingDataFolder):
        self.trainingData = trainingData                    # training data name
        self.testingDataFolder = testingDataFolder          # testing data folder name
        self.totalClsNum = 12
        self.featureNum = 3
        self.filterWinSize = 51

    def getclassname(self, clsNum):
        # Output strings of class name

        # Parameters
        # ------
        # clsNum :       class number
        
        # Returns
        # ------
        #     class names

        if clsNum == 0:
            clsname = 'Bench Press'
        elif clsNum == 1:
            clsname = 'Shoulder Press'
        elif clsNum == 2:
            clsname = 'Bicep Curls'
        elif clsNum == 3:
            clsname = 'Upright Rows'
        elif clsNum == 4:
            clsname = 'Lateral Raises'
        elif clsNum == 5:
            clsname = 'Overhead Triceps Extensions'
        elif clsNum == 6:
            clsname = 'Kneeling Triceps Kickbacks'
        elif clsNum == 7:
            clsname = 'Standing Bent-over Rows'
        elif clsNum == 8:
            clsname = 'Kneeling Bent-over Rows'
        elif clsNum == 9:
            clsname = 'Squats'
        elif clsNum == 10:
            clsname = 'Forward Walking Lunges'
        elif clsNum == 11:
            clsname = 'Calf Raises'
        return clsname


    def smooth(self, rawSig, winSize, winType='hanning'):
        # Low Pass Filter Function

        # Parameters
        # ------
        # rawSig:          input signal                                        
        # winSize:         dimension of smoothing window; should be odd number      
        # winType:         window type including 'flat', 'hanning', 'hamming', 'bartlett' 'blackman'
        
        # Returns
        # ------
        # Low pass filtered data in list
        
        sig = np.squeeze(np.asarray(rawSig)) 		# convert matrix to array
        #print sig.shape
        processedSig = np.r_[2*sig[0]-sig[winSize-1::-1],sig,2*sig[-1]-sig[-1:-winSize:-1]]
        #print processedSig.shape
        if winType == 'flat':
            w = np.ones(winSize, 'd')
        else:
            w = eval('np.' + winType + '(winSize)')
        smoothed = np.convolve(w/w.sum(), processedSig, mode='same')
        smoothedSig = smoothed[winSize:-winSize+1]
        smoothedSig = np.array(smoothedSig).tolist() 	# convert array to list
        return smoothedSig


    def readInData(self, filename, clsNumber):
        # Read in testing data 

        # Parameters
        # ------
        # filename :         data file name
        
        # Returns
        # ------
        #    matrix containing time, X, Y, Z and label

        ###

        # input file check
        try:
            dataFile = open(filename, 'r+')
        except IOError:
            print 'Could not open the testing data file!'

        # Initilization
        time = []; x = []; y = []; z = []; clsNum = []
        
        for line in dataFile.readlines():
            res = line.split(',')
            if clsNumber == int(res[-1]): 			# check the label
                #tmp = res[0].split(':')
                #time.append(float(tmp[1])) 			# Time list 
                time.append(float(res[0]))
                x.append(float(res[1]))
                y.append(float(res[2]))
                z.append(float(res[3]))
                clsNum.append(int(res[4]))
        dataFile.close()
        dataMat = np.matrix([time, x, y, z, clsNum]) 	# noisy signal matrix
        filteredDataMat = np.zeros((dataMat.shape[0], dataMat.shape[1])) # create empty filtered data matrix 
        filteredDataMat[0,:] = dataMat[0,:] 		# time in noisy signal and low-passed are the same 
        for i in range(1, 4):
            filteredDataMat[i, :] = self.smooth(dataMat[i, :], 91)
        return (filteredDataMat, dataMat)                   # 5 x colNum matrices


    def findLargestVar(self, data):
        # Find the largest variance among X, Y and Z

        # Parameters
        # ------
        # data :             5 x colNum matrix
        
        # Returns
        # ------
        #       Axis that has the largest variance

        varCoord = []
        for i in range(1, 4):
            varCoord.append(abs(np.var(data[i, :])))
        return varCoord.index(max(varCoord))


    def detectRange(self, smoothed):
        # Detect whether the data is in certain range or not

        # Parameters
        # ------
        # data :              5 x colNum matrix
        
        # Returns
        # ------
        #       indices of range within two sigma of the mean

        mean = np.mean(smoothed)
        var = np.var(smoothed)
        twoSigRange = mean + 1.5*var
        return twoSigRange


    def detectPeaks(self, x, mpd=120, threshold=0, edge='rising'):
        # Peaks Detection Function

        # Parameters
        #----------
        # x : 1D array_like
        #    data.
        # mpd : positive integer, optional (default = 1)
        #    detect peaks that are at least separated by minimum peak distance (in
        #    number of data).

        # Returns
        #-------
        #    indeces of the peaks in `x`.

        varIdx = self.findLargestVar(x)
        smoothed = np.array(x[varIdx+1, :])
        cleanedTime = np.array(self.shiftTime(x[0, :]))
        
        smoothed = np.atleast_1d(smoothed).astype('float64')
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
        
        peakIdx = []
        for item in ind:
            if smoothed[item] > self.detectRange(smoothed):
                peakIdx.append(item)
        peakList = smoothed[peakIdx]
        timeList = cleanedTime[peakIdx]
        return (peakList, timeList, peakIdx)


    def moments(self, dataMat):
        # Compute moments from data and make sure that all the moments are at the same order

        # Parameters
        # ------
        # dataMat :         5 x colNum matrix

        # Returns
        # ------
        #       3x4 matrix containing four moments

        # Feature selection
        m1 = np.mean(dataMat[1:4,], axis=1)
        m1 = np.matrix(m1).reshape(3, 1)
        m2 = np.var(dataMat[1:4,], axis=1)
        m2 = np.power(m2, 1.0/2)
        m2 = np.matrix(m2).reshape(3, 1)
        m3 = moment(dataMat[1:4,], 3, axis=1)
        m3 = np.power(abs(m3), 1.0/3)*np.sign(m3)
        m3 = np.matrix(m3).reshape(3, 1)
        m4 = moment(dataMat[1:4,], 4, axis=1)
        m4 = np.power(abs(m4), 1.0/4)
        m4 = np.matrix(m4).reshape(3, 1)
        momentMat = np.concatenate((m1, m2, m3, m4), axis=1)
        #print momentMat
        return momentMat


    def shiftTime(self, tmMatrix):
        # Shift time interval to [0: end of interval]

        # Parameters
        # ------
        # tm :         time list

        # Returns
        # ------
        #       shifited time array starting from 0

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


    def plot(self, dataMat, clsNum, filename):
        # Plot function

        # Parameters
        # ------
        # dataMat :    data matrix
        # clsNum :     class number

        # Returns
        # ------
        #       plots of original data, low pass filtered data and peak selection

        peakList, timeList, peakIdx = self.detectPeaks(dataMat, mpd=90)
        return len(peakList)
        # plot raw data
        plt.figure()
        plt.title(filename + ' ' + classname(clsNum) + ' : ' + str(clsNum))
        plt.xlabel('Time')
        plt.ylabel('Acceleration Data')
        sx, = plt.plot(self.shiftTime(dataMat[0,:]), dataMat[1,:], 'r-')  
        sy, = plt.plot(self.shiftTime(dataMat[0,:]), dataMat[2,:], 'b-')
        sz, = plt.plot(self.shiftTime(dataMat[0,:]), dataMat[3,:], 'g-')
        plt.plot(timeList, peakList, 'k*')
        plt.legend([sx, sy, sz], ['X', 'Y', 'Z'], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size':8})
        plt.grid(True)
        #plt.savefig(filename + '_' + classname(clsNum) + '.png')
        #plt.show()
        return len(peakList)


    def featureExtract(self, dataMat, clsNum):
        # Feature Extraction 
        
        # Parameters
        # ------
        # dataMat :          5 x colNum data matrix

        # Returns
        # ------
        #          1x13 feature vector containing normalized moments

        tmpFeatMat = self.moments(dataMat)
        featMat = tmpFeatMat.reshape(1, 12)
        tmpFeatArr = np.squeeze(np.array(featMat))
        featArr = np.append(tmpFeatArr, clsNum)
        #print featArr
        return featArr


    def classify(self, trainingFeat, testingFeat):
        # classify the testing data by measuring cos theta = A.*B / (||A|| * ||B||)

        # Parameters
        # ------
        # trainingFeat :           training feature matrix
        # testingFeat :            testing feature matrix

        # Returns
        # ------
        #     classification results from 0 to 11

        res = []
        for i in range(trainingFeat.shape[0]):
            meanSim = np.dot(trainingFeat[i][0:-1], testingFeat[0:-1])/(scialg.norm(trainingFeat[i][0:-1], 2)*scialg.norm(testingFeat[0:-1], 2))
            #print meanSim
            #stdSim = np.dot(trainingFeat[i][3:6], testingFeat[3:6])/(scialg.norm(trainingFeat[i][3:6], 2)*scialg.norm(testingFeat[3:6], 2))
            res.append(meanSim)
        return res.index(max(res))


    def calcAccuracy(self, clsRes, testingDataLabel):
        # Calculate classification accuracy

        # Parameters
        # ------
        # testingData :         testing matrix

        # Returns
        # ------
        #     classification accuracy

        cntCorrect = 0
        wrongCls = []
        for i in range(len(clsRes)):
            if clsRes[i] == testingDataLabel[i]:
               cntCorrect += 1
            else:
               if (clsRes[i],testingDataLabel[i]) not in wrongCls:
                  wrongCls.append((clsRes[i], testingDataLabel[i]))
        res = float(cntCorrect) / len(clsRes)
        return (res, wrongCls)


    def training(self, trainingData):
        # Training stage

        # Parameters
        # ------
        # trainingData :            raw training data

        # Returns
        # ------
        #      12 x 13 training data feature matrix

        print 'Training '
        trainingFeatRowNum = 12; trainingFeatColNum = 13
        trainingFeatMat = np.zeros((trainingFeatRowNum, trainingFeatColNum))  # 12 x 13 feature matrix
        tolClsNum = 12         # total class number 
        # Training
        for clsNum in range(tolClsNum):
            # Read in data
            filteredTrainSig, noisyTrainSig = self.readInData(trainingData, clsNum)    # 5 x colNum matrices
            assert(filteredTrainSig.shape[0] == 5), "Row number of the training data is wrong"
            time.sleep(.1)
            print '\b. ', 
            sys.stdout.flush()
            trainingFeat = self.featureExtract(filteredTrainSig, clsNum)                  # 1 x 12 vector 
            trainingFeatMat[clsNum, ] = trainingFeat
        print '\nDone!'
        return trainingFeatMat


    def testing(self, testingData, trainingFeatMat):
        # Show testing results

        # Parameters
        # ------
        # testingData :                   raw testing data
        # trainingFeatMat :               training feature matrix

        # Returns
        # ------
        #      testing results 

        # testing features
        clsResList = []; testLabel = []
        tolClsNum = 12
        reps = []
       
        for clsNum in range(tolClsNum):
            filteredTestSig, noisyTestSig = self.readInData(testingData, clsNum)
            assert(filteredTestSig.shape[0] == 5), "Row number of the testing data is wrong"
            testingFeatArr = self.featureExtract(filteredTestSig, clsNum)                  # 1 x 12 vector
            reps.append(self.plot(filteredTestSig, clsNum, testingData[-11:-4]))
            testLabel.append(int(testingFeatArr[-1]))   # testing data labels  
            clsRes = self.classify(trainingFeatMat, testingFeatArr)
            clsResList.append(clsRes)
        #print reps
        return (clsResList, testLabel, reps)
        

    def showResults(self, clsResList, testLabel):
        # Show classification results
        
        # Parameters
        # ------
        # clsResList :            classification results list
        # testLabel  :            testing data label

        # Returns
        # ------
        #       display results

        accuracy, wrongCls = self.calcAccuracy(clsResList, testLabel)
        print "Classification Accuracy is %.2f%s."  %(accuracy * 100, '%')
        print '\n'
        tolClsNum = 12
        clsResArr = np.array(clsResList); testArr = np.array(testLabel)
        confusionMat = np.zeros((tolClsNum, tolClsNum))
        for i in range(tolClsNum):
            for j in range(tolClsNum):
                confusionMat[i,j] = np.sum(np.where(clsResArr==i, 1, 0)*np.where(testArr==j, 1, 0))
        print "Confusion Matrix is: \n"
        print confusionMat


    def cosineSimilarityClf(self):
        # Main Classification function
        
        # Parameters
        # ------

        # Returns
        # ------
        #       Show classification accuracy

        # Checking arguments
        assert(len(sys.argv) == 3), 'Please type two arguments: training data name, testing data folder name'
        # Training
        trainingFeatMat = self.training(self.trainingData)
        # Testing
        result = []; label = []
        os.chdir(self.testingDataFolder)
        print 'Classifying ... '
        totalReps = [0] * 12
        totalReps = np.array(totalReps)
        for testingData in glob.glob('*.csv'):
            try:
                open(testingData, 'r')
            except IOError:
                print 'Could not open the file ' + testingData + '\n'
            time.sleep(.1)
            print '\t ' + testingData 
            sys.stdout.flush()
            resList, testLabel, reps = self.testing(testingData, trainingFeatMat)
            result += resList; label += testLabel
            totalReps += np.array(reps)
        print totalReps
        print 'Done!'
        # Show Results
        self.showResults(result, label)


#------------------------------------------------------------#
#                 Run it from here                           #
#------------------------------------------------------------#
if __name__ == "__main__":
    exercise = ResistanceClassification(sys.argv[1], sys.argv[2])
    exercise.cosineSimilarityClf();
