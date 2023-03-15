import numpy as np
from scipy.io import wavfile
import pyximport; pyximport.install()
from os import path
import time
from scipy.ndimage.filters import gaussian_filter1d
from tflite_runtime import interpreter as inter

import nms as nms
from spectrogram import compute_features_spectrogram


class NeuralNet:

    def __init__(self, params_):
        """
        Creates a CNN for features computation and an SVM or XGBoost model for detection and classification.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """

        self.params = params_
        self.network_features = None
        self.model_feat = None
        self.network_classif = None

    def test(self, goal, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):
        """
        Makes a prediction on the position, probability and class of the calls present in an audio file.
        
        Parameters
        -----------
        goal : String
            Indicates whether the file needs to be tested for detection or classification.
            Can be either "detection" or "classification".
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

        # compute features
        tic = time.time()
        features = self.create_or_load_features(goal, file_name, audio_samples, sampling_rate)
        toc=time.time()
        self.params.features_computation_time += toc - tic
        features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1).astype("float32")
        tic = time.time()

        # perform detection
        print(features.shape)
        feat_test = self.model_feat.predict(features)
        toc=time.time()
        self.params.detect_time += toc - tic
        
        # perform classification
        tic = time.time()
        y_predictions = self.network_classif.predict_proba(feat_test)
        toc=time.time()
        self.params.classif_time += toc - tic

        # smooth the output prediction per column so smooth each class prediction over time
        tic = time.time()
        if self.params.smooth_op_prediction:
            y_predictions = gaussian_filter1d(y_predictions, self.params.smooth_op_prediction_sigma, axis=0)
        
        # trying to get rid of rows with 0 highest
        call_predictions_bat = y_predictions[:,1:]
        call_predictions_not_bat = y_predictions[:,0]
        high_preds = np.array([np.max(x) for x in call_predictions_bat])[:, np.newaxis]
        pred_classes = np.array([np.argmax(x)+1 for x in call_predictions_bat])[:, np.newaxis]
        
        # perform non max suppression
        pos, prob, pred_classes, call_predictions_not_bat = nms.nms_1d(high_preds[:,0].astype(np.float),
                                    pred_classes, call_predictions_not_bat, self.params.nms_win_size, file_duration)

        # remove pred that have a higher probability of not being a bat
        pos_bat = []
        prob_bat = []
        pred_classes_bat = []
        for i in range(len(pos)):
            if prob[i][0]>call_predictions_not_bat[i]:
                pos_bat.append(pos[i])
                prob_bat.append(prob[i])
                pred_classes_bat.append(pred_classes[i])
        toc=time.time()
        self.params.nms_computation_time += toc - tic

        nms_pos = np.array(pos_bat)
        nms_prob = np.array(prob_bat)
        pred_classes = np.array(pred_classes_bat)
        nb_windows = features.shape[0]
        return  nms_pos, nms_prob, pred_classes, nb_windows

    def create_or_load_features(self, goal, file_name=None, audio_samples=None, sampling_rate=None):
        """
        Does 1 of 3 possible things
        1) computes feature from audio samples directly
        2) loads feature from disk OR
        3) computes features from file name

        Parameters
        -----------
        goal : String
            Indicates whether the features are used for detection or classification.
            Can be either "detection" or "classification".
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
        elif goal =="classification":
            audio_dir = self.params.audio_dir_classif
            data_set = self.params.data_set_classif

        # 1) computes feature from audio samples directly
        if file_name is None:
            features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
        else:
            # 2) loads feature from disk
            if self.params.load_features_from_file and path.exists(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1]  + '_spectrogram' + '.npy'):
                features = np.load(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram'  + '.npy')
            # 3) computes features from file name
            else:
                if self.params.load_features_from_file: print("missing features have to be computed ", self.params.feature_dir + data_set + '_' + file_name.split("/")[-1]  + '_spectrogram' )
                #sampling_rate, audio_samples = wavfile.read("20200806_230000T.wav")
                sampling_rate, audio_samples = wavfile.read(audio_dir + file_name.split("/")[-1]  + '.wav')
                features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
                if self.params.save_features_to_file or self.params.load_features_from_file:
                    np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram', features)
        return features

    def save_features(self, goal, files):
        """
        Computes and saves features to disk.

        Parameters
        ----------
        goal : String
            Indicates whether the features are computed for detection or classification.
            Can be either "detection" or "classification".
        files : String
            Name of the wav file used to make a prediction.
        """
        
        if goal == "detection":
            audio_dir = self.params.audio_dir_detect
            data_set = self.params.data_set_detect
        elif goal =="classification":
            audio_dir = self.params.audio_dir_classif
            data_set = self.params.data_set_classif

        for file_name in files:
            sampling_rate, audio_samples = wavfile.read(audio_dir + file_name.split("/")[-1] + '.wav')
            features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
            np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1]  + '_spectrogram', features)
