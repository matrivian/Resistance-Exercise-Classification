"""
 RESISTANCE TRAINING CLASSIFICATION: 
 Function: classify testing data into one of the 12 classes
           and count how many exercises repeated

 Input: Training Date and Testing Data
 Output: Testing accuracy and repeated times

 Author:  Jun Guo
 Date:    12/24/2015

 License: GPL3.0
"""

import numpy as np
import csv
import os
import glob
import time
import sys
import scipy.linalg as scialg
import matplotlib.pyplot as plt
from scipy.stats.mstats import moment
from sklearn import svm
from sklearn import metrics


class PreProcessing:
    """ Pre-processing raw data """

    def __init__(self, win_size):
        self.win_size = win_size      # window size of the low-pass filter

    def lpf(self, raw_data, win_size, win_type='hanning'):
        """
        Low-pass filter

        raw_data :  input data
        win_size :  window size, the bigger the number, the smoother the filtered signal
        win_type :  'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        return : low-pass filtered data
        """
        data = np.squeeze(np.asarray(raw_data)) 		# convert matrix to array
        processed = np.r_[2 * data[0] - data[win_size - 1::-1],
                          data, 2 * data[-1] - data[-1:-win_size:-1]]
        if win_type == 'flat':
            w = np.ones(win_size, 'd')
        else:
            w = eval('np.' + win_type + '(win_size)')
        smoothed = np.convolve(w / w.sum(), processed, mode='same')
        filtered = smoothed[win_size:-win_size + 1]
        # convert array to list
        filtered = np.array(filtered).tolist()
        return filtered

    def shift_time(self, time_vec):
        """
        Shift time to [0: end of interval] e.g., some dataset has time 
        vector like [59.0, 59.1, ..., 0.0, 0.1, ..., 10.0], after shifting time,
        time vector should start from 0 to the end 

        time_vec : original time list
        return : modified time vector
        """
        shifted_time = list(np.array(time_vec).reshape(-1,))
        cycle = 60
        end = 59.9
        if (shifted_time[-1] - shifted_time[0]) > 0:
            shift = shifted_time[0]
            for i in range(len(shifted_time)):
                shifted_time[i] -= shift
        elif (shifted_time[-1] - shifted_time[0]) < 0:
            shift = shifted_time[0]
            for i in range(len(shifted_time)):
                if shifted_time[i] >= shift and shifted_time[i] <= end:
                    shifted_time[i] -= shift
                elif shifted_time[i] < shift:
                    shifted_time[i] += cycle
                    shifted_time[i] -= shift
        return shifted_time


class PeakDetection:
    """ Count exercise repetitions by detecting peaks of signals """

    def __init__(self, data, tm_axis):
        self.data = data            # the dimension of data is row_num x 3
        self.tm_axis = tm_axis      # time axis

    def find_largest_var(self, mat):
        """
        Find the largest variance among X, Y and Z

        mat : data matrix 
        Return: the index of largest variance
        """
        var_coord = [abs(np.var(mat[:, i])) for i in range(3)]
        return var_coord.index(max(var_coord))

    def detect_range(self, data_vec):
        """
        Detect whether the data is in certain range or not

        data_vec : one row or column of the data matrix
        Return: the range within 1.5 variance
        """
        mean = np.mean(data_vec)
        var = np.var(data_vec)
        sig_range = mean + 1.5 * var
        return sig_range

    def detect_peaks(self, x, mpd=120, threshold=0):
        """
        Peaks Detection Function

        x : 1D array_like data
        mpd : positive integer detect peaks that are at least separated by 
                minimum peak distance (in number of data)
        threshold : peaks - neighbors threshold 
        Return : peaks lists, peaks time list, peaks indices 
        """
        var_idx = self.find_largest_var(self.data)
        smoothed = np.array(self.data[:, var_idx])
        smoothed = np.atleast_1d(smoothed).astype('float64')
        # print smoothed
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
            ind = ind[np.in1d(ind, np.unique(
                np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == smoothed.size - 1:
            ind = ind[:-1]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack(
                [smoothed[ind] - smoothed[ind - 1], smoothed[ind] - smoothed[ind + 1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(smoothed[ind])][
                ::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (smoothed[ind[i]] > smoothed[ind])
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their
            # occurrence
            ind = np.sort(ind[~idel])

        peak_idx = [item for item in ind if smoothed[
            item] > self.detect_range(smoothed)]
        peak_lst = smoothed[peak_idx]
        time_lst = self.tm_axis[peak_idx]
        return (peak_lst, time_lst, peak_idx)


class FeatureExtraction():
    """ Feature Extraction """

    def __init__(self, axises_num):
        self.axises_num = axises_num
        self.moments_num = 4

    def calc_moments(self, data, ex_dict):
        """
        Calculate four moments of one exercise

        data : input matrix and the dimension is rows number x 3
        return : feature matrix 4x3 where, 4 : moments; 3 : x, y, z
        """
        m1 = np.mean(data, axis=0)
        m2 = np.var(data, axis=0)
        m2 = np.power(m2, 1.0 / 2)
        m3 = moment(data, 3, axis=0)
        m3 = np.power(abs(m3), 1.0 / 3) * np.sign(m3)
        m4 = moment(data, 4, axis=0)
        m4 = np.power(abs(m4), 1.0 / 4)
        ft_mat = np.concatenate((m1, m2, m3, m4), axis=0)
        ft_mat = ft_mat.reshape(self.moments_num, self.axises_num)
        return ft_mat

    def get_ft_nn(self, filename, ex_dict):
        """
        Get feature matrix for nearest neighbor

        filename : data file name
        ex_dict : resistance exercises of 12 classes
        return : feature matrix (12x12) and compress data labels
                 and compressed labels 1x12
        """
        ft_mat = np.zeros((1, self.axises_num * self.moments_num))
        tm, data, label = load_data(
            filename, ex_dict, plot=False, cnt_reps=False)
        cpr_label = []
        for ex_num in range(len(ex_dict)):
            ind = label == ex_num
            tmp_ft = self.calc_moments(data[ind, :], ex_dict)
            tmp_ft = tmp_ft.reshape(1, self.axises_num * self.moments_num)
            ft_mat = np.vstack((ft_mat, tmp_ft))
            tmp_label = [ex_num] * tmp_ft.shape[0]
            cpr_label += [elem for elem in tmp_label]
        ft_mat = np.delete(ft_mat, (0), axis=0)
        return ft_mat, cpr_label

    def get_ft_svm(self, filename, ex_dict):
        """
        Get feature matrix for svm

        filename : data file name
        ex_dict : resistance exercises of 12 classes
        return : feature matrix (48x3) and compress data labels
                 and compressed labels 1x48
        """
        ft_mat = np.zeros((1, self.axises_num))
        tm, data, label = load_data(
            filename, ex_dict, plot=False, cnt_reps=False)
        cpr_label = []
        for ex_num in range(len(ex_dict)):
            ind = label == ex_num
            tmp_ft = self.calc_moments(data[ind, :], ex_dict)
            ft_mat = np.vstack((ft_mat, tmp_ft))
            tmp_label = [ex_num] * self.moments_num
            cpr_label += [elem for elem in tmp_label]
        ft_mat = np.delete(ft_mat, (0), axis=0)
        return ft_mat, cpr_label


class NearestNeighbor:
    """ 
    Classification of resistance exercises using nearest neighbor
    the distance function is cosine similarity or dot product
    """

    def __init__(self, rt_exercises, axises_num):
        self.rt = rt_exercises
        self.axises_num = axises_num

    def training(self, tr_filename):
        """
        NN training function

        tr_filename : training file name
        return : training set feature 12x12
        """
        print 'NN Training ... '
        ft = FeatureExtraction(self.axises_num)
        tr_ft, tr_cpr_label = ft.get_ft_nn(tr_filename, self.rt)
        print 'Done.'
        return tr_ft

    def predict_one_class(self, tr_ft, te_ft):
        """
        Compare similarity between testing set feature and training set feature 

        tr_ft : training feature
        te_ft : testing feature
        return : which class (0-11) does the te_ft belong to
        """
        score = []
        for i in range(tr_ft.shape[0]):
            sim_score = calc_similarity(tr_ft[i], te_ft)
            score += [sim_score]
        return score.index(max(score))

    def predict_one_file(self, tr_ft, te_filename, ex_dict):
        """
        Predict one testing file 

        tr_ft : training feature
        te_filename : testing filename 
        ex_dict : resistance exercises dictionary
        return : predicted label and actual label for a testing set
        """
        est_label = []
        ft = FeatureExtraction(self.axises_num)
        te_ft, te_cpr_label = ft.get_ft_nn(te_filename, ex_dict)
        for ex_num in range(len(ex_dict)):
            tmp_res = self.predict_one_class(tr_ft, te_ft[ex_num])
            est_label += [tmp_res]
        return est_label, te_cpr_label

    def predict_all(self, tr_filename, te_folder):
        """
        Classify the testing data in a folder

        tr_filename : training dataset file name
        te_folder : testing dataset folder name
        ex_dict : resistance exercises dictionary
        return : repetitions and predicted labels
        """
        tr_ft = self.training(tr_filename)        # training
        est_label, te_label = [], []
        os.chdir(te_folder)
        print 'NN Classifying ... '                  # classifying
        for te_filename in glob.glob('*.csv'):
            time.sleep(.01)
            print '\t ' + te_filename
            sys.stdout.flush()
            est_label_one, te_label_one = self.predict_one_file(
                tr_ft, te_filename, self.rt)
            est_label += est_label_one
            te_label += range(len(self.rt))
        print 'Done.'
        est_label = np.array(est_label)
        res = Results()
        res.calc_rt_accuracy(te_label, est_label)
        res.calc_confusion_mat(te_label, est_label, self.rt)


class SvmRTClf:
    """Classification of resistance exercises using suppor vector machine"""

    def __init__(self, rt_exercises, axises_num):
        self.rt = rt_exercises
        self.axises_num = axises_num

    def predict_all(self, foldername):
        """
        Classification use svm

        foldername : folder which contains all data
        return : classification accuracy and confusion matrix
        """
        all_te_label, est_label = [], []
        ft = FeatureExtraction(self.axises_num)
        res = Results()
        for te_filename in glob.glob(foldername + '*.csv'):
            # Leave-one-out as testing
            te_ft, te_label = ft.get_ft_svm(te_filename, self.rt)
            # all training features
            all_tr_ft = np.zeros((1, self.axises_num))
            # all the rest as training
            print 'SVM Training ... '
            all_tr_label = []
            for tr_filenames in glob.glob(foldername + '*.csv'):
                if tr_filenames != te_filename:
                    one_tr_ft, one_tr_label = ft.get_ft_svm(
                        tr_filenames, self.rt)
                    all_tr_ft = np.vstack((all_tr_ft, one_tr_ft))
                    all_tr_label += one_tr_label
            all_tr_ft = np.delete(all_tr_ft, (0), axis=0)
            # SVM training
            C = 1.0           # SVM regularization parameter
            clfr = svm.SVC(kernel='rbf', gamma=300, C=C)
            clfr.fit(all_tr_ft, all_tr_label)
            print 'Done Training.'
            # SVM prediction
            print 'SVM Classifying ... '
            clf_res = clfr.predict(te_ft)
            # print clf_res
            est_label += res.convert_clf_result(clf_res, self.rt)
            print 'Done Classifying ' + te_filename[-11:] + '\n'
            all_te_label += range(12)
        est_label = np.array(est_label)
        res.calc_rt_accuracy(all_te_label, est_label)
        res.calc_confusion_mat(all_te_label, est_label, self.rt)


class Results:
    """ show results """

    def convert_clf_result(self, res, ex_dict):
        """
        Convert SVM classification result to a vector contain 0-11

        res : SVM predicted results vector 1x48
        ex_dict : resistance exercises
        return : reduced dimention vector 1x12
        """
        ans = []
        st, end, interval = 0, 4, 4
        for _ in range(len(ex_dict)):
            cnt = np.bincount(res[st:end].astype(int))
            if max(cnt) == 1:
                ans.append(res[st].astype(int))
            else:
                ind = np.argmax(cnt)
                ans.append(ind)
            st += interval
            end += interval
        return ans

    def calc_rt_accuracy(self, te_label, est_label):
        """
        calculate resistance exercises classification accuracy

        te_label : acutal testing data label vector
        est_label : predicted label vector
        """
        ans = np.mean(est_label == te_label)
        ans = ans * 100
        print 'Classification accuracy : %.2f%%' % ans

    def calc_confusion_mat(self, te_label, est_label, ex_dict):
        """
        compute confusion matrix

        te_label : acutal testing data label vector
        est_label : predicted label vector
        """
        actual = np.array(te_label)
        est = np.array(est_label)
        n = len(ex_dict)
        conf_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                conf_mat[i, j] = np.sum(
                    np.where(est == i, 1, 0) * np.where(actual == j, 1, 0))
        print 'Confusion Matrix is: \n'
        print conf_mat


def load_data(filename, ex_dict, plot=False, cnt_reps=False):
    """
    Load one file into three arrays, time, data, label then filter the data.  
    Each csv file has 5 columns: 1st : time sample; 2nd : X axis; 
    3rd : Y axis; 4th : Z axis; 5th : exercise label

    filename : data file name
    ex_dict : resistance exercises dictionary
    plot : change this to True if plot wanted
    cnt_reps : change this to True if counting repetitions of the exercises wanted

    return : time, raw data, label vectors and filtered data
    """
    time, label = [], []
    raw_data = np.array([0] * 3)
    with open(filename, 'r+') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            time += [float(row[0])]
            raw_data = np.vstack((raw_data, map(float, row[1:-1])))
            label += [int(row[-1])]
    time, label = np.array(time), np.array(label)
    raw_data = np.delete(raw_data, (0), axis=0)

    # initialize the filter window size as 91
    prep = PreProcessing(91)
    filtered = np.zeros(raw_data.shape)
    for i in range(3):
        # low-pass filter the raw data
        filtered[:, i] = prep.lpf(raw_data[:, i], prep.win_size)
    # tr_time = prep.shift_time(tr_tm)   # shift time
    if plot:
        plot_ex(filename, time, raw_data, filtered,
                label, ex_dict, plot_peak=True)
    if cnt_reps:
        cnt = count_reps(time, filtered, label, ex_dict)
        print cnt
    return time, filtered, label


def plot_ex(filename, tm, raw_data, filtered, label, ex_dict, plot_peak=False):
    """
    Plot exercises figures 

    tm : time 
    raw_data : original un-filtered data
    filtered : low pass filtered data
    label : label vector
    ex_dict : resistance exercise dictionary
    plot_peak : change this to True if peaks are wanted in the plot
    """
    print 'Plotting ... '
    for ex_num in range(len(ex_dict)):
        ind = label == ex_num
        plt.figure()
        time.sleep(.01)
        print '\t. ' + ex_dict[ex_num]
        sys.stdout.flush()
        # raw data
        plt.subplot(211)
        plt.title(ex_dict[ex_num])
        plt.xlabel('Time (s)')
        plt.ylabel('Raw Acc Data (g)')
        plt.xlim([-0.5, max(list(tm[ind])) + 0.5])
        rx, = plt.plot(tm[ind], raw_data[ind, 0], 'r-', marker='o',
                       markersize=8, markevery=(len(raw_data[ind, 0]) - 1) / 3, lw=1.5)
        ry, = plt.plot(tm[ind], raw_data[ind, 1], 'g-', marker='v',
                       markersize=8, markevery=(len(raw_data[ind, 1]) - 1) / 3, lw=1.5)
        rz, = plt.plot(tm[ind], raw_data[ind, 2], 'b-', marker='d',
                       markersize=8, markevery=(len(raw_data[ind, 2]) - 1) / 3, lw=1.5)
        plt.legend([rx, ry, rz], ['X', 'Y', 'Z'], bbox_to_anchor=(
            1.01, 1), loc=2, borderaxespad=0., prop={'size': 8})
        plt.grid(True)
        # filtered data
        plt.subplot(212)
        plt.xlabel('Time (s)')
        plt.ylabel('Raw Acc Data (g)')
        maxX, maxY, maxZ = max(filtered[ind, 0]), max(
            filtered[ind, 1]), max(filtered[ind, 2])
        minX, minY, minZ = min(filtered[ind, 0]), min(
            filtered[ind, 1]), min(filtered[ind, 2])
        plt.ylim([min(minX, minY, minZ) - 0.1, max(maxX, maxY, maxZ) + 0.1])
        plt.xlim([-0.5, max(list(tm[ind])) + 0.5])
        fx, = plt.plot(tm[ind], filtered[ind, 0], 'r-', marker='o',
                       markersize=8, markevery=(len(filtered[ind, 0]) - 1) / 3, lw=1.5)
        fy, = plt.plot(tm[ind], filtered[ind, 1], 'g-', marker='v',
                       markersize=8, markevery=(len(filtered[ind, 1]) - 1) / 3, lw=1.5)
        fz, = plt.plot(tm[ind], filtered[ind, 2], 'b-', marker='d',
                       markersize=8, markevery=(len(filtered[ind, 2]) - 1) / 3, lw=1.5)
        # plot peaks for data
        if plot_peak:
            pd = PeakDetection(filtered[ind, :], tm[ind])
            pk_arr, tm_arr, pk_ind = pd.detect_peaks(filtered[ind, :])
            plt.plot(tm_arr, pk_arr, 'k*', markersize=12)
        plt.legend([fx, fy, fz], ['X', 'Y', 'Z'], bbox_to_anchor=(
            1.01, 1), loc=2, borderaxespad=0., prop={'size': 8})
        plt.grid(True)
        plt.savefig(filename[:-4] + '_' + ex_dict[ex_num] + '.png')
    print 'Done Plotting.'


def count_reps(time_vec, data, label, ex_dict):
    """
    Count repetitions using peak detection

    time_vec : time sample vector
    data : low-pass filtered data
    label : label of data
    ex_dict : resistance exercises dictionary
    return : reps for each class
    """
    reps = np.zeros(len(ex_dict))
    for ex_num in range(len(ex_dict)):
        ind = label == ex_num
        pd = PeakDetection(data[ind, :], time_vec[ind])
        pk_arr, tm_arr, pk_ind = pd.detect_peaks(data[ind, :])
        reps[ex_num] = len(pk_arr)
    return reps


"""
Calculate similarity using a*b/(||a||_{2}*||b||_{2})
"""
calc_similarity = lambda a, b: np.dot(
    a, b) / (scialg.norm(a, 2) * scialg.norm(b, 2))


def resistance_clf(clfr):
    """
    Classification of resistance exercises

    nn_clf : nearest neighbor using cosine similarity 
    smv_clf : support vector machine using rbf kernel
    """

    rt = {0: 'Bench_Press', 1: 'Shoulder_Press', 2: 'Bicep_Curls', 3: 'Upright_Rows',
          4: 'Lateral_Raises', 5: 'Overhead_Triceps_Extensions', 6: 'Kneeling_Triceps_Kickbacks',
          7: 'Standing_Bent-over_Rows', 8: 'Kneeling_Bent-over-Rows', 9: 'Squats',
          10: 'Forward_Walking_Lunges', 11: 'Calf_Raises'}

    axises_num = 3

    if clfr == 'nn':
        nn = NearestNeighbor(rt, axises_num)
        nn.predict_all(sys.argv[2], sys.argv[3])

    elif clfr == 'svm':
        svm_rt_clf = SvmRTClf(rt, axises_num)
        svm_rt_clf.predict_all(sys.argv[2])

    else:
        raise AssertionError('Please type either nn or svm!') 

if __name__ == '__main__':
    resistance_clf(sys.argv[1])
