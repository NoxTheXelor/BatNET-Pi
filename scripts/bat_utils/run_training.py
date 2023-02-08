import numpy as np
import tensorflow as tf
import json
import time
import joblib
import gc
import os

from bat_utils.models_params_helper import params_to_dict
import bat_utils.evaluate as evl
from bat_utils.data_set_params import DataSetParams
import bat_utils.classifier as clss

def load_data(data_set, goal):
    """
    Loads the training and testing positions, files, durations and classes into separate arrays.

    Parameters
    -----------
    data_set : String
        Path to the npz dataset.
    goal : String
        Indicates whether the file needs to be tested for detection or classification.
        Can be either "detection" or "classification".
    
    Returns
    --------          
    train_pos : ndarray
        Ground truth positions of the calls for each training file.
    train_files : numpy array
        Names of the wav files used to train the model.  
    train_durations : numpy array
        Durations of the wav files used to train the model. 
    train_classes : numpy array
        Ground truth class for each training file.
    test_pos : ndarray
        Ground truth positions of the calls for each test file.
    test_files : numpy array
        Names of the wav files used to test the model.  
    test_durations : numpy array
        Durations of the wav files used to test the model. 
    test_classes : numpy array
        Ground truth class of the calls for each test file.
    """

    loaded_data_tr = np.load(data_set, fix_imports=True, allow_pickle=True, encoding = 'latin1')
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
    type_train = None; type_test = None
    i = 0; j = 0
    while type_train is None:
        type_train = None if len(train_pos[i])==0 else type(train_pos[i][0])
        i += 1
    while type_test is None:
        type_test = None if len(test_pos[j])==0 else type(test_pos[j][0])
        j += 1
    if type_train!=np.ndarray:
        for ii in range(len(train_pos)):
            train_pos[ii] = train_pos[ii][..., np.newaxis]
    if type_test!=np.ndarray:
        for ii in range(len(test_pos)):
            test_pos[ii] = test_pos[ii][..., np.newaxis]

    # In case of detection, batdetective's npz is used and the filenames need to be decoded.
    if goal == "detection":
        train_files = np.array(list(map(lambda x : x.decode('ascii'), train_files)))
        test_files = np.array(list(map(lambda x : x.decode('ascii'), test_files)))
    elif goal == "classification":
        test_classes = loaded_data_tr['test_class']
        train_classes = loaded_data_tr['train_class']
        # Some datasets like Batdetective's put an additional axis around each class while others do not.
        # We uniformise by adding the axis when it is not present.
        type_train = None; type_test = None
        i = 0; j = 0
        while type_train is None:
            type_train = None if len(train_classes[i])==0 else type(train_classes[i][0])
            i += 1
        while type_test is None:
            type_test = None if len(test_classes[j])==0 else type(test_classes[j][0])
            j += 1
        if type_train!=np.ndarray:
            for ii in range(len(train_classes)):
                train_classes[ii] = train_classes[ii][..., np.newaxis]
        if type_test!=np.ndarray:
            for ii in range(len(test_classes)):
                test_classes[ii] = test_classes[ii][..., np.newaxis]

    return train_pos, train_files, train_durations, train_classes, test_pos, test_files, test_durations, test_classes

def save_model(model_name, model, model_dir, threshold_classes):
    """
    Saves the model and the parameters.

    Parameters
    -----------
    model_name : String
        Name of the model.
    model : Classifier
        The model that needs to be saved.
    model_dir : String
        Path to the directory in which the model will be saved.
    threshold_classes : numpy array
        Thresholds used to evaluate the performance of the model.
    """

    print("\_".join(model.params.model_identifier_classif.split("_")))

    if model_name in ["cnn2", "hybrid_call_svm", "hybrid_call_xgboost"]: # cnn detect
        model.model.network_detect.save(model_dir + model.params.model_identifier_detect + '_model')
        weights = model.model.network_detect.get_weights()
        np.save(model_dir + model.params.model_identifier_detect + '_weights', weights)
    if model_name in ["batmen", "cnn2"]: # cnn classif
        model.model.network_classif.save(model_dir + model.params.model_identifier_classif + '_model')
        weights = model.model.network_classif.get_weights()
        np.save(model_dir + model.params.model_identifier_classif + '_weights', weights)
    if model_name in ["hybrid_cnn_svm", "hybrid_cnn_xgboost"]: # cnn features
        model.model.network_features.save(model_dir + model.params.model_identifier_features + '_model')
        weights = model.model.network_features.get_weights()
        np.save(model_dir + model.params.model_identifier_features + '_weights', weights)
    if model_name in ["hybrid_cnn_svm", "hybrid_call_svm"]: # svm
        joblib.dump(model.model.network_classif, model_dir + model.params.model_identifier_classif + '_model.pkl')
        joblib.dump(model.model.scaler, model_dir + model.params.model_identifier_classif + '_scaler.pkl')
    if model_name in ["hybrid_cnn_xgboost","hybrid_call_xgboost"]: # xgboost
        joblib.dump(model.model.network_classif, model_dir + model.params.model_identifier_classif + '_model.pkl')

    mod_params = params_to_dict(model.params)
    misc_params = {'win_size':0, 'chunk_size':0, 'max_freq':0, 'min_freq':0, 'mean_log_mag':0, 'slice_scale':0, 'overlap':0,
                'crop_spec':False, 'denoise':False, 'smooth_spec':False, 'nms_win_size':0, 'smooth_op_prediction_sigma':0, 'num_hnm':0}
    misc_params['win_size'] = model.model.params.window_size
    misc_params['max_freq'] = model.model.params.max_freq
    misc_params['min_freq'] = model.model.params.min_freq
    misc_params['mean_log_mag'] = model.model.params.mean_log_mag
    misc_params['slice_scale'] = model.model.params.fft_win_length
    misc_params['overlap'] = model.model.params.fft_overlap
    misc_params['crop_spec'] = model.model.params.crop_spec
    misc_params['denoise'] = model.model.params.denoise
    misc_params['smooth_spec'] = model.model.params.smooth_spec
    misc_params['nms_win_size'] = model.model.params.nms_win_size
    misc_params['smooth_op_prediction_sigma'] = model.model.params.smooth_op_prediction_sigma
    misc_params['num_hnm'] = model.model.params.num_hard_negative_mining
    mod_params.update(misc_params)
    print("mod_params=",mod_params)

    params_file = model_dir + model.params.model_identifier_classif + '_params.p'
    with open(params_file, 'w') as fp:
        json.dump(mod_params, fp)
    
    params_file_txt = model_dir + model.params.model_identifier_classif + '_perf_params.txt'
    with open(params_file_txt, 'a') as fp:
        fp.write(str(mod_params))
    
    np.save(model_dir + model.params.model_identifier_classif + '_thresholds.npy', threshold_classes)

def delete_models(model_name, model):
    """
    Deletes the models to free memory.

    Parameters
    -----------
    model_name : String
        Name of the model.
    model : Classifier
        The model that needs to be deleted.
    """

    if model_name in ["batmen", "cnn2", "hybrid_cnn_svm", "hybrid_cnn_xgboost", "hybrid_call_svm", "hybrid_call_xgboost"]: # cnn detect
        tf.keras.backend.clear_session()
        gc.collect()
    if model_name in ["hybrid_cnn_xgboost","hybrid_call_xgboost"]:
        for clf in model.model.network_classif.estimators_:
            clf._Booster.__del__()
    gc.collect()

if __name__ == '__main__':
    """
    This can be used to run and evaluate different algorithms for bat echolocation classification.
    The results can vary by a few percent from run to run.
    """

    ####################################
    # Parameters to be set by the user #
    ####################################
    on_GPU = True   # True if tensorflow runs on GPU, False otherwise
    # the name of the datasets used for detection, classification and validation
    test_set_detect = ''
    test_set_classif = ''
    validation_set = ''
    # the path to the npz files used for detection, classification and validation
    data_set_detect = ''
    data_set_classif = ''
    data_set_valid = ''
    # the path to the directories containing the detection, classification and validation audio files
    raw_audio_dir_detect = ''
    raw_audio_dir_classif = ''
    raw_audio_dir_valid = ''
    # the path to the directories in which the results, the models and the features will be saved
    result_dir = 'results/'
    model_dir = 'data/models/'
    feature_dir = 'data/features/'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(feature_dir):
        os.mkdir(feature_dir)
    model_name = "hybrid_cnn_xgboost" # one of: batmen, cnn2, hybrid_cnn_svm, hybrid_cnn_xgboost,
    # hybrid_call_svm, hybrid_call_xgboost

    if on_GPU:
        # needed to run tensorflow on GPU
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        # needed to run tensorflow on CPU
        config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
        tf.config.set_visible_devices([], 'GPU')
        session = tf.compat.v1.InteractiveSession(config=config)

    print("\nModel:", model_name)
    print('test set detect:', test_set_detect)
    print('test set classif:', test_set_classif)

    tic = time.time()
    # load data for detection, classification and validation
    train_pos_detect, train_files_detect, train_durations_detect, train_classes_detect, \
        _, test_files_detect, _, _ = load_data(data_set_detect, "detection")
    train_pos_classif, train_files_classif, train_durations_classif, train_classes_classif, \
        test_pos_classif, test_files_classif, test_durations_classif, test_classes_classif = load_data(data_set_classif, "classification")
    _, train_files_valid, _, _, pos_valid, files_valid, durations_valid, classes_valid = load_data(data_set_valid, "classification")

    # load parameters
    params = DataSetParams(model_name)
    params.classification_model = model_name
    params.time = time.strftime("%d_%m_%y_%H_%M_%S_")
    params.model_identifier_detect = params.time + "detect_" + params.classification_model
    params.model_identifier_features = params.time + "features_" + params.classification_model
    params.model_identifier_classif = params.time + "classif_" + params.classification_model
    params.audio_dir_detect = raw_audio_dir_detect
    params.audio_dir_classif = raw_audio_dir_classif
    params.audio_dir_valid = raw_audio_dir_valid
    params.model_dir = model_dir
    params.feature_dir = feature_dir
    params.data_set_detect = test_set_detect
    params.data_set_classif = test_set_classif
    params.data_set_valid = validation_set
    params.load_params_from_csv(model_name)
    model = clss.Classifier(params)

    # train and test
    if model_name in ['batmen',  'hybrid_cnn_svm', 'hybrid_cnn_xgboost']:
        model.train(train_files_classif, train_pos_classif, train_durations_classif, train_classes_classif,
                    files_valid, pos_valid, durations_valid, classes_valid, test_files_classif, test_pos_classif,
                    test_durations_classif, test_classes_classif)
        nms_pos, nms_prob, pred_classes, nb_windows = model.test_batch("classification", test_files_classif, test_durations_classif)
    elif model_name in ['cnn2', 'hybrid_call_svm', 'hybrid_call_xgboost']:
        train_pos = {"detect": train_pos_detect, "classif": train_pos_classif}
        train_files = {"detect": train_files_detect, "classif": train_files_classif}
        train_durations = {"detect": train_durations_detect, "classif": train_durations_classif}
        train_classes = {"detect": train_classes_detect, "classif": train_classes_classif}
        model.train(train_files, train_pos, train_durations, train_classes, files_valid, pos_valid, durations_valid, classes_valid,
                    test_files_classif, test_pos_classif, test_durations_classif, test_classes_classif)
        nms_pos, nms_prob, pred_classes, nb_windows = model.test_batch("classification", test_files_classif, test_durations_classif)
    session.close()
    
    # tune the thresholds on the validation set and evaluate the performance on the test set
    nms_pos_valid, nms_prob_valid, pred_classes_valid, nb_windows_valid = model.test_batch("classification", files_valid, durations_valid)
    threshold_classes = evl.prec_recall_1d( nms_pos_valid, nms_prob_valid, pos_valid, pred_classes_valid,
                        classes_valid, durations_valid, model.params.detection_overlap, model.params.window_size, 
                        nb_windows_valid, model_dir + model.params.model_identifier_classif + '_perf_params.txt',
                        True)
    evl.prec_recall_1d( nms_pos, nms_prob, test_pos_classif, pred_classes, test_classes_classif, 
                        test_durations_classif, model.params.detection_overlap, model.params.window_size, 
                        nb_windows, model_dir + model.params.model_identifier_classif + '_perf_params.txt',
                        False, threshold_classes=threshold_classes)
    
    # save the model
    save_model(model_name, model, model_dir, threshold_classes)

    toc = time.time()
    time_no_features = toc-tic-model.params.features_computation_time
    total_time = toc-tic
    print('features computation time', round(model.params.features_computation_time, 3), '(secs) =', round((model.params.features_computation_time)/60,2), r"min \\")
    print('run time without features', round(time_no_features, 3), '(secs) =', round((time_no_features)/60,2), r"min \\")
    print('total run time', round(total_time, 3), '(secs) =', round((total_time)/60,2), r"min \\")
    with open(model_dir + model.params.model_identifier_classif + '_perf_params.txt', 'a') as f:
        f.write('features computation time '+ str(round(model.params.features_computation_time, 3))+ ' (secs) = '+ str(round((model.params.features_computation_time)/60,2))+ " min \n")
        f.write('run time without features '+ str(round(time_no_features, 3))+ ' (secs) = '+ str(round((time_no_features)/60,2))+ " min \n")
        f.write('total run time '+ str(round(total_time, 3))+ ' (secs) = '+ str(round((total_time)/60,2))+ " min \n")
    
    delete_models(model_name,model)