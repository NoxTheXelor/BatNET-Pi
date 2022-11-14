import numpy as np
from scipy.io import wavfile
import pyximport; pyximport.install()
from os import path
import time
import gc
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.utils import class_weight
from sklearn.svm import SVC
import xgboost as xgb

import nms_cnn2 as nms
from spectrogram import compute_features_spectrogram
from call_features import compute_features_call
from cnn_helper import network_fit, tune_network
from svm_xgb_helper import tune_svm, tune_xgb
from models_params_helper import params_to_dict


class NeuralNet:

    def __init__(self, params_):
        """
        Creates a CNN for detection and an SVM or XGBoost model for classification.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """
        self.params = params_
        self.network_detect = None
        self.network_classif = None
        self.scaler = None

    def train(self, positions, class_labels, files, durations):
        """
        Takes the file names and ground truth call positions and trains model.

        Parameters
        -----------
        positions : ndarray
            Training positions for each training file.
        class_labels : numpy array
            Class label for each training position.
        files : numpy array
            Names of the wav files used to train the model.        
        durations : numpy array
            Durations of the wav files used to train the model.
        """

        # free memory
        if self.params.classification_model == "hybrid_cnn_xgboost" and self.network_classif is not None:
            for clf_estimator in self.network_classif.estimators_:
                clf_estimator._Booster.__del__()
            tf.keras.backend.clear_session()
            gc.collect()
        
        # compute or load the features of the training files and the associated class label.
        print("Compute or load features")
        tic = time.time()
        features_detect, labels_detect, _ = self.features_labels_from_file(positions["detect"], class_labels["detect"], files["detect"],
                                                                        durations["detect"], "detection")
        features_classif, labels_classif, _ = self.features_labels_from_file(positions["classif"], class_labels["classif"], files["classif"],
                                                                        durations["classif"], "classification")
        toc = time.time()
        self.params.features_computation_time += toc-tic
        
        # tuning of the hyperparameters of the detection CNN and fit the model       
        if self.params.tune_cnn_2:
            print("Tune CNN detect")
            tic_cnn_2 = time.time()
            tune_network(self.params, features_detect, labels_detect, labels_detect, 
                        self.params.trials_filename_1, goal="detection")
            toc_cnn_2 = time.time()
            while toc_cnn_2-tic_cnn_2 < self.params.tune_time:
                tune_network(self.params, features_detect, labels_detect, labels_detect, 
                        self.params.trials_filename_1, goal="detection")
                toc_cnn_2 = time.time()
            print('total tuning time', round(toc_cnn_2-tic_cnn_2, 3), '(secs) =', round((toc_cnn_2-tic_cnn_2)/60,2), r"min \\")
        
        self.network_detect, _ = network_fit(self.params, features_detect, labels_detect, 
                                                        labels_detect, 2)
                
        # train and tune the svm classification model
        if self.params.classification_model == "hybrid_call_svm":
            self.scaler = MinMaxScaler()
            features_classif = self.scaler.fit_transform(features_classif)
            if self.params.tune_svm_call:
                print("Tune SVM")
                tic_svm = time.time()
                tune_svm(self.params, features_classif, labels_classif, self.params.trials_filename_2)
                toc_svm = time.time()
                while toc_svm-tic_svm < self.params.tune_time:
                    tune_svm(self.params, features_classif, labels_classif, self.params.trials_filename_2)
                    toc_svm = time.time()
                print('total tuning time', round(toc_svm-tic_svm, 3), '(secs) =', round((toc_svm-tic_svm)/60,2), r"min \\")
            
            print("Fit SVM")
            tic = time.time()
            self.network_classif = MultiOutputClassifier(SVC( kernel=self.params.kernel, C=self.params.C, degree=self.params.degree,
                                        gamma=self.params.gamma_svm, class_weight=self.params.class_weight,
                                        probability=True, verbose=False, max_iter=self.params.max_iter), n_jobs=-1)
            print("SVM best params =",params_to_dict(self.params))
            self.network_classif.fit(features_classif, labels_classif)
            toc = time.time()
            print('total SVM run time', round(toc-tic, 3), '(secs) =', round((toc-tic)/60,2), r"min \\")
        
        # train and tune the xgb classification model
        elif self.params.classification_model == "hybrid_call_xgboost":
            if self.params.tune_xgboost_call:
                print("Tune xgboost")
                tic_xgb = time.time()
                tune_xgb(self.params, features_classif, labels_classif, self.params.trials_filename_2)
                toc_xgb = time.time()
                while toc_xgb-tic_xgb < self.params.tune_time:
                    tune_xgb(self.params, features_classif, labels_classif, self.params.trials_filename_2)
                    toc_xgb = time.time()
                print('total tuning time', round(toc_xgb-tic_xgb, 3), '(secs) =', round((toc_xgb-tic_xgb)/60,2), r"min \\")
            
            print("Fit xgboost")
            tic = time.time()
            self.network_classif = MultiOutputClassifier(xgb.XGBClassifier(eta=self.params.eta,
                                                    min_child_weight=self.params.min_child_weight,
                                                    max_depth=self.params.max_depth, n_estimators=self.params.n_estimators,
                                                    gamma=self.params.gamma_xgb, subsample=self.params.subsample,
                                                    scale_pos_weight=self.params.scale_pos_weight, objective="binary:logistic",
                                                    tree_method='gpu_hist'))       
            class_w = class_weight.compute_sample_weight('balanced',labels_classif)
            self.network_classif.fit(features_classif, labels_classif, sample_weight=class_w)
            toc = time.time()
            print("XGBoost best params =",params_to_dict(self.params))
            print('total xgboost run time', round(toc-tic, 3), '(secs) =', round((toc-tic)/60,2), r"min \\")
            
    
    def features_labels_from_file(self, positions, class_labels, files, durations, goal):
        """
        Computes or loads the features of each position of the files
        and indicates the associated class label.

        Parameters
        -----------
        positions : ndarray
            Training positions for each file.
        class_labels : numpy array
            Class label for each position.
        files : numpy array
            Names of the wav files.        
        durations : numpy array
            Durations of the wav files.
        goal : String
            Indicates whether the network needs to be tuned for detection or classification.
            Can be either "detection" or "classification".

        Returns
        --------
        features : ndarray
            Array containing the spectrogram features for each training position of the audio files.
        labels : ndarray
            Class labels in one-hot encoding for each training position of the audio files.
        labels_not_merged : ndarray
            Array containing one class label per call instead of per position in one-hot encoding.
            (Used to compute the class weights.)
        """

        feats = []
        labels = np.array([])
        labels_not_merged = np.array([], dtype=int)
        nb_inds_no_dup = 0
        for i, file_name in enumerate(files):
            if positions[i].shape[0] > 0:
                local_feats = self.create_or_load_features(goal, file_name)

                # convert time in file to integer
                positions_ratio = positions[i] / durations[i]
                train_inds = (positions_ratio*float(local_feats.shape[0])).astype('int')

                if goal=="detection":
                    feats.append(local_feats[train_inds, :, :, :])
                    labels = np.concatenate((labels,class_labels[i]))
                elif goal == "classification":
                    # one-hot encoding of the class labels
                    local_class = np.zeros((class_labels[i].size, 7), dtype=int)
                    rows = np.arange(class_labels[i].size)
                    local_class[rows, class_labels[i]-1] = 1
                    train_inds_no_dup = []

                    # combine call pos that are in the same window and merge their labels
                    for pos_ind, win_ind  in enumerate(train_inds):
                        # if the pos to add is in a new window then add it
                        if pos_ind==0 or train_inds_no_dup[-1]!=win_ind:
                            train_inds_no_dup.append(win_ind)
                            if pos_ind==0 and labels.shape[0]==0: labels = np.array([local_class[pos_ind]])
                            else: labels = np.concatenate((labels,np.array([local_class[pos_ind]])), axis=0)
                        else:
                            index_one = np.where(local_class[pos_ind]==1)[0][0]
                            # if the pos to add is in the same window but it is a new class then combine the labels
                            # with all entries of the same window
                            if labels[-1][index_one]!=1:
                                same_win_ind = np.where(train_inds_no_dup==win_ind)[0] + nb_inds_no_dup
                                labels[same_win_ind] = np.logical_or(labels[same_win_ind],local_class[pos_ind]).astype('int')
                            # if the pos to add is in the same window and it is not a new class then add it
                            # only if it is the first class that was observed for that window (to generate duplicates)
                            elif labels[-1].sum() == 1:
                                train_inds_no_dup.append(win_ind)
                                labels = np.concatenate((labels,np.array([local_class[pos_ind]])), axis=0)
                    feats.append(local_feats[train_inds_no_dup, :])
                    if labels_not_merged.shape[0] == 0: labels_not_merged = local_class
                    else: labels_not_merged = np.vstack((labels_not_merged, local_class))
                    nb_inds_no_dup += len(train_inds_no_dup)
        
        if goal=="detection": labels = labels.astype(np.uint8)
        features = np.vstack(feats)
        return features, labels, labels_not_merged

    def test(self, goal, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):
        """
        Makes a prediction on the position, probability and class of the calls present in an audio file.
        
        Parameters
        -----------
        goal : String
            Indicates whether the features are used for detection or classification
            or more specifically for validation.
            Can be either "detection", "classification" or "validation".
        file_name : String
            Name of the wav file used to make a prediction.
        file_duration : float
            Duration of the wav file used to make a prediction.
        audio_samples : numpy array
            Data read from wav file.
        sampling_rate : int
            Sample rate of wav file.

        Returns
        --------
        nms_pos : ndarray
            Predicted positions of calls for every test file.
        nms_prob : ndarray
            Confidence level of each prediction for every test file.
        pred_classes : ndarray
            Predicted class of each prediction for every test file.
        nb_windows : ndarray
            Number of windows for every test file.
        """

        # compute features and perform detection
        tic = time.time()
        features_detect = self.create_or_load_features(goal, file_name, audio_samples, sampling_rate, "_spectrogram")
        toc=time.time()
        self.params.features_computation_time += toc-tic
        features_detect = features_detect.reshape(features_detect.shape[0], features_detect.shape[2], features_detect.shape[3], 1)
        nb_windows = features_detect.shape[0]
        tic = time.time()  
        y_predictions_detect = self.network_detect.predict(features_detect)
        toc=time.time()
        self.params.detect_time += toc - tic

        # compute features for classification
        tic = time.time()
        features_classif = self.create_or_load_features(goal, file_name, audio_samples, sampling_rate, "_call")
        toc=time.time()
        self.params.features_computation_time += toc-tic
        if self.params.classification_model == "hybrid_call_svm":
            features_classif = self.scaler.transform(features_classif)
        

        # smooth the output prediction per column so smooth each class prediction over time
        tic = time.time()
        if self.params.smooth_op_prediction:
            y_predictions_detect = gaussian_filter1d(y_predictions_detect, self.params.smooth_op_prediction_sigma, axis=0)
        
        # trying to get rid of rows with 0 highest
        call_predictions_bat = y_predictions_detect[:,1:]
        call_predictions_not_bat = y_predictions_detect[:,0]
        high_preds = np.array([np.max(x) for x in call_predictions_bat])[:, np.newaxis]
        pred_classes = np.array([np.argmax(x)+1 for x in call_predictions_bat])[:, np.newaxis]
        
        # perform non max suppression
        pos, prob, pred_classes, call_predictions_not_bat, features_classif = nms.nms_1d(high_preds[:,0].astype(np.float), pred_classes, 
                                                                    call_predictions_not_bat, features_classif, self.params.nms_win_size, file_duration)
  
        # remove pred that have a higher probability of not being a bat
        pos_bat = []
        prob_bat = []
        pred_classes_bat = []
        features_bat = []
        for i in range(len(pos)):
            if prob[i][0]>call_predictions_not_bat[i]:
                pos_bat.append(pos[i])
                prob_bat.append(prob[i])
                pred_classes_bat.append(pred_classes[i])
                features_bat.append(features_classif[i])
        
        toc=time.time()
        self.params.nms_computation_time += toc-tic

        # perform classification
        tic = time.time()
        pred_proba = np.array([])
        pred_classes = np.array([])
        if np.array(features_bat).shape[0] != 0:
            y_predictions_classif = self.network_classif.predict_proba(np.array(features_bat))
            y_predictions_classif = np.array(y_predictions_classif)[:,:,1].T
            pred_proba = y_predictions_classif.flatten('F')[..., np.newaxis]
            pred_classes = np.repeat(np.arange(1,8,1),len(pos_bat))
        toc=time.time()
        self.params.classif_time += toc - tic

        nms_pos = np.array(pos_bat*7)
        nms_prob = pred_proba
        return nms_pos, nms_prob, pred_classes, nb_windows

    def create_or_load_features(self, goal, file_name=None, audio_samples=None, sampling_rate=None, feature_type=None):
        """
        Does 1 of 3 possible things
        1) computes feature from audio samples directly
        2) loads feature from disk OR
        3) computes features from file name

        Parameters
        -----------
        goal : String
            Indicates whether the features are used for detection or classification
            or more specifically for validation.
            Can be either "detection", "classification" or "validation".
        file_name : String
            Name of the wav file used to make a prediction.
        audio_samples : numpy array
            Data read from wav file.
        sampling_rate : int
            Sample rate of wav file.

        Returns
        --------
        features : ndarray
            Array containing the spectrogram features for each window of the audio file.
        """

        if goal == "detection":
            audio_dir = self.params.audio_dir_detect
            data_set = self.params.data_set_detect
            feat_type = '_spectrogram'
        elif goal =="classification":
            audio_dir = self.params.audio_dir_classif
            data_set = self.params.data_set_classif
            feat_type = '_call'
        if feature_type is not None:
            feat_type = feature_type

        # 1) computes feature from audio samples directly
        if file_name is None:
            features= compute_features(goal, audio_samples, sampling_rate, self.params, feat_type)
        else:
            # 2) loads feature from disk
            if self.params.load_features_from_file and path.exists(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + feat_type + '.npy'):
                features = np.load(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + feat_type + '.npy')
            # 3) computes features from file name
            else:
                if self.params.load_features_from_file: print("missing features have to be computed")
                sampling_rate, audio_samples = wavfile.read(audio_dir + file_name.split("/")[-1]  + '.wav')
                features = compute_features(goal, audio_samples, sampling_rate, self.params, feat_type)
                if self.params.save_features_to_file or self.params.load_features_from_file:
                    np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + feat_type, features)
        return features

    def save_features(self, goal, files):
        """
        Computes and saves features to disk.

        Parameters
        ----------
        goal : String
            Indicates whether the features are used for detection or classification
            or more specifically for validation.
            Can be either "detection", "classification" or "validation".
        files : String
            Name of the wav file used to make a prediction.
        """

        audio_dir_detect = self.params.audio_dir_detect
        data_set_detect = self.params.data_set_detect
        audio_dir_classif = self.params.audio_dir_classif
        data_set_classif = self.params.data_set_classif
        audio_dir_valid = self.params.audio_dir_valid
        data_set_valid = self.params.data_set_valid
        for file_name in files:
            if goal == "detection":
                sampling_rate, audio_samples = wavfile.read(audio_dir_detect + file_name.split("/")[-1] + '.wav')
                features = compute_features(goal, audio_samples, sampling_rate, self.params)
                np.save(self.params.feature_dir + data_set_detect + '_' + file_name.split("/")[-1] + '_spectrogram', features)
            elif goal == "classification":
                sampling_rate, audio_samples = wavfile.read(audio_dir_classif + file_name.split("/")[-1] + '.wav')
                features = compute_features(goal, audio_samples, sampling_rate, self.params)
                np.save(self.params.feature_dir + data_set_classif + '_' + file_name.split("/")[-1] + '_call', features)
            elif goal == "validation":
                sampling_rate, audio_samples = wavfile.read(audio_dir_valid + file_name.split("/")[-1] + '.wav')
                features = compute_features(goal, audio_samples, sampling_rate, self.params)
                np.save(self.params.feature_dir + data_set_valid + '_' + file_name.split("/")[-1] + '_call', features)
    

def compute_features(goal, audio_samples, sampling_rate, params, feat_type=None):
    """
    Computes either spectrograms or call features in function of the goal and the feature type.

    Parameters
    -----------
    goal : String
        Indicates whether the features are used for detection or classification
        or more specifically for validation.
        Can be either "detection", "classification" or "validation".
    audio_samples : numpy array
        Data read from wav file.
    sampling_rate : int
        Sample rate of wav file.
    params : DataSetParams
        Parameters of the model.
    feat_type : String
        "_call" if the type of features to be computed are call features
        "_spectrogram" if the type of features to be computed are spectrograms
    
    Returns
    --------
    features : ndarray
        Array containing the spectrogram or call features for each window of the audio samples.
    """

    if feat_type=="_spectrogram" or (feat_type is None and goal == "detection"):
        temp = compute_features_spectrogram(audio_samples, sampling_rate, params)
        return temp
    elif feat_type=="_call" or goal == "classification" or goal=="validation":
        temp =  compute_features_call(audio_samples, sampling_rate, params)
        return temp
    else:
        return None

