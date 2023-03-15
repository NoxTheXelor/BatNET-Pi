import numpy as np
import pickle
import os
import time
from sklearn.datasets import load_files
from tflite_support.metadata_writers import writer_utils
import json
import joblib
from data_set_params import DataSetParams
from larq_compute_engine.tflite.python import interpreter
import classifier as clss
import xgboost as xgb
import tflite
from tflite_runtime import interpreter as inter


def load_data(data_set, goal):
    loaded_data_tr = np.load(data_set, fix_imports=True, allow_pickle=True, encoding='latin1')
    train_pos = loaded_data_tr['train_pos']
    train_files = loaded_data_tr['train_files']
    train_durations = loaded_data_tr['train_durations']
    train_classes = []
    test_pos = loaded_data_tr['test_pos']
    test_files = loaded_data_tr['test_files']
    test_durations = loaded_data_tr['test_durations']
    test_classes = []

    # Some datasets like Batdetective's put an additional axis around each position while others do not.
    # We uniformise by adding the axis when it is not present.
    type_train = None
    type_test = None
    i = 0
    j = 0
    while type_train is None:
        type_train = None if len(train_pos[i]) == 0 else type(train_pos[i][0])
        i += 1
    while type_test is None:
        type_test = None if len(test_pos[j]) == 0 else type(test_pos[j][0])
        j += 1
    if type_train != np.ndarray:
        for ii in range(len(train_pos)):
            train_pos[ii] = train_pos[ii][..., np.newaxis]
    if type_test != np.ndarray:
        for ii in range(len(test_pos)):
            test_pos[ii] = test_pos[ii][..., np.newaxis]

    if goal == "detection":
        train_files = np.array(list(map(lambda x: x.decode('ascii'), train_files)))
        test_files = np.array(list(map(lambda x: x.decode('ascii'), test_files)))
    elif goal == "classification":
        test_classes = loaded_data_tr['test_class']
        train_classes = loaded_data_tr['train_class']
        # Some datasets like Batdetective's put an additional axis around each class while others do not.
        # We uniformise by adding the axis when it is not present.
        type_train = None
        type_test = None
        i = 0
        j = 0
        while type_train is None:
            type_train = None if len(train_classes[i]) == 0 else type(train_classes[i][0])
            i += 1
        while type_test is None:
            type_test = None if len(test_classes[j]) == 0 else type(test_classes[j][0])
            j += 1
        if type_train != np.ndarray:
            for ii in range(len(train_classes)):
                train_classes[ii] = train_classes[ii][..., np.newaxis]
        if type_test != np.ndarray:
            for ii in range(len(test_classes)):
                test_classes[ii] = test_classes[ii][..., np.newaxis]
    print("train size", goal, " = ", train_files.shape)
    print("test size", goal, " = ", test_files.shape)
    return train_pos, train_files, train_durations, train_classes, test_pos, test_files, test_durations, test_classes


if __name__ == "__main__":
    load_features_from_file = False
    model_name = "hybrid_cnn_xgboost"  # can be one of: 'batmen', 'cnn2',  'hybrid_cnn_svm', 'hybrid_cnn_xgboost', 'hybrid_call_svm', 'hybrid_call_xgboost'
    result_dir = 'results/'  # where we will store the outputs
    #model_dir = 'model_raspberry/' # Binary model
    model_dir = 'raspberry_model_V2/' # Binary model with another XGBoost model using 500 estimators
    #model_dir = "model_float/" # Float model

    test_set = 'uk'
    data_set_test = '../data/train_test_split/test_set_' + test_set + '.npz'
    raw_audio_dir_detect = '../data/wav/'
    raw_audio_dir_classif = '../data/wav/'

    _, _, _, _, pos_test, files_test, durations_test, classes_test = load_data(data_set_test, "detection")

    # model name and load model
    #date = "04_03_22_17_59_02_" # Binary model
    date = "01_05_22_21_14_57_" # Second XGBoost model
    #date = "14_04_22_10_57_47_" # Float model
    hnm = ""
    model_file_features = model_dir + date + "features_" + model_name + hnm
    #network_features = load_model(model_file_features + '_model')
    network_feat = interpreter.Interpreter(writer_utils.load_file(model_dir + "raspberry_model.tflite", mode='rb'), num_threads=4)
    model_file_classif = model_dir + date + "classif_" + model_name + hnm
    network_classif = xgb.XGBClassifier()
    network_classif.load_model(model_file_classif + '_model.json')

    # load params
    with open(model_file_classif + '_params.p') as f:
        parameters = json.load(f)
    print("params=", parameters)

    # array with group name according to class number
    group_names = ['not call', 'Barbarg', 'Envsp', 'Myosp', 'Pip35', ' Pip50', 'Plesp', 'Rhisp']

    # model classifier
    params = DataSetParams(model_name)
    params.window_size = parameters['win_size']
    params.max_freq = parameters['max_freq']
    params.min_freq = parameters['min_freq']
    params.mean_log_mag = parameters['mean_log_mag']
    params.fft_win_length = parameters['slice_scale']
    params.fft_overlap = parameters['overlap']
    params.crop_spec = parameters['crop_spec']
    params.denoise = parameters['denoise']
    params.smooth_spec = parameters['smooth_spec']
    params.nms_win_size = parameters['nms_win_size']
    params.smooth_op_prediction_sigma = parameters['smooth_op_prediction_sigma']
    params.n_estimators = parameters["n_estimators"]
    params.load_features_from_file = load_features_from_file
    params.detect_time = 0
    params.classif_time = 0
    params.audio_dir_detect = raw_audio_dir_detect
    params.audio_dir_classif = raw_audio_dir_classif
    model_cls = clss.Classifier(params)

    model_cls.model.network_classif = network_classif
    model_cls.model.network_features = None
    model_cls.model.model_feat = network_feat

    nms_pos_test, nms_prob_test, pred_classes_test, nb_windows_test = model_cls.test_batch("classification", files_test,
                                                                                           durations_test)

    print("Features computation time " + str(params.features_computation_time))
    print("Detection time " + str(params.detect_time))
    print("Classification time " + str(params.classif_time))
    print("NMS computation time " + str(params.nms_computation_time))
