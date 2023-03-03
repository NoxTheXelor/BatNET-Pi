from pathlib import Path
from tzlocal import get_localzone
import datetime
import sqlite3
import requests
import json
import time
import math
import numpy as np
import librosa
import operator
import socket
import threading
import os

from utils.notifications import sendAppriseNotifications
from utils.parse_settings import config_to_settings

from bat_utils import write_op as wo
from bat_utils import classifier as clss
from bat_utils.data_set_params import DataSetParams
from tensorflow.keras.models import Model, load_model
from bat_utils.run_classifier import read_audio, run_classifier

from scipy.io import wavfile
import numpy as np
import os
import glob
import time
import tensorflow as tf
import json
from tensorflow.keras.models import Model, load_model
import joblib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''



HEADER = 64
PORT = 5050
SERVER = "localhost"
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

userDir = os.path.expanduser('~')
DB_PATH = userDir + '/BirdNET-Pi/scripts/birds.db'


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    server.bind(ADDR)
except BaseException:
    print("Waiting on socket")
    time.sleep(5)


# Open most recent Configuration and grab DB_PWD as a python variable
with open('scripts/thisrun.txt', 'r') as f:
    this_run = f.readlines()
    audiofmt = "." + str(str(str([i for i in this_run if i.startswith('AUDIOFMT')]).split('=')[1]).split('\\')[0])
    priv_thresh = float("." + str(str(str([i for i in this_run if i.startswith('PRIVACY_THRESHOLD')]).split('=')[1]).split('\\')[0])) / 10


"""def loadModel():

    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    model_name = "hybrid_cnn_xgboost"
    date = "02_06_21_09_32_04_"
    hnm_iter = "2"
    model_dir= "models\\"
    model_file_features = model_dir + date + "features_" + model_name + "_hnm" + hnm_iter
    network_features = load_model(model_file_features + '_model')
    network_feat = Model(inputs=network_features.input, outputs=network_features.layers[-3].output)
    model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm_iter
    network_classif = joblib.load(model_file_classif + '_model.pkl')

        # load params
    with open(model_file_classif + '_params.p') as f:
        parameters = json.load(f)
    print("params=", parameters)


    # Load labels
    CLASSES = []
    CLASSES.append('not call')
    CLASSES.append('Barbarg')
    CLASSES.append('Envsp')
    CLASSES.append('Myosp')
    CLASSES.append('Pip35')
    CLASSES.append('Pip50')
    CLASSES.append('Plesp')
    CLASSES.append('Rhisp')

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
    if model_name in ["hybrid_cnn_xgboost", "hybrid_call_xgboost"]: params.n_estimators = parameters["n_estimators"]
    params.load_features_from_file = False
    params.detect_time = 0
    params.classif_time = 0
    model_cls = clss.Classifier(params)
    if model_name in  ["batmen", "cnn2", "hybrid_cnn_svm", "hybrid_cnn_xgboost", "hybrid_call_svm", "hybrid_call_xgboost"]:
        model_cls.model.network_classif = network_classif
    if model_name in ["cnn2", "hybrid_call_svm", "hybrid_call_xgboost"]:
        model_cls.model.network_detect = network_detect
    if model_name in ["hybrid_cnn_svm", "hybrid_cnn_xgboost"]:
        model_cls.model.network_features = network_features
        model_cls.model.model_feat = network_feat
    if model_name in ["hybrid_cnn_svm", "hybrid_call_svm"]:
        model_cls.model.scaler = scaler



    print('DONE!')

    return myinterpreter
"""

def loadCustomSpeciesList(path):

    slist = []
    if os.path.isfile(path):
        with open(path, 'r') as csfile:
            for line in csfile.readlines():
                slist.append(line.replace('\r', '').replace('\n', ''))

    return slist


def splitSignal(sig, rate, overlap, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp

        sig_splits.append(split)

    return sig_splits


def readAudioData(path, overlap, sample_rate=48000):

    print('READING AUDIO DATA...', end=' ', flush=True)

    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')

    # Split audio into 3-second chunks
    chunks = splitSignal(sig, rate, overlap)

    print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    return chunks


def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])


def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))


def predict(sample, sensitivity):
    global INTERPRETER
    # Make a prediction
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    INTERPRETER.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    INTERPRETER.invoke()
    prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

#     # print("DATABASE SIZE:", len(p_sorted))
#     # print("HUMAN-CUTOFF AT:", int(len(p_sorted)*priv_thresh)/10)
#
#     # Remove species that are on blacklist

    human_cutoff = max(10, int(len(p_sorted) * priv_thresh))

    for i in range(min(10, len(p_sorted))):
        if p_sorted[i][0] == 'Human_Human':
            with open(userDir + '/BirdNET-Pi/HUMAN.txt', 'a') as rfile:
                rfile.write(str(datetime.datetime.now()) + str(p_sorted[i]) + ' ' + str(human_cutoff) + '\n')

    return p_sorted[:human_cutoff]


def analyzeAudioData(chunks, lat, lon, week, sensitivity, overlap,):
    global INTERPRETER

    detections = {}
    start = time.time()
    print('ANALYZING AUDIO...', end=' ', flush=True)

    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)

    # Parse every chunk
    pred_start = 0.0
    for c in chunks:

        # Prepare as input signal
        sig = np.expand_dims(c, 0)

        # Make prediction
        p = predict([sig, mdata], sensitivity)
#        print("PPPPP",p)
        HUMAN_DETECTED = False

        # Catch if Human is recognized
        for x in range(len(p)):
            if "Human" in p[x][0]:
                HUMAN_DETECTED = True

        # Save result and timestamp
        pred_end = pred_start + 3.0

        # If human detected set all detections to human to make sure voices are not saved
        if HUMAN_DETECTED is True:
            p = [('Human_Human', 0.0)] * 10

        detections[str(pred_start) + ';' + str(pred_end)] = p

        pred_start = pred_end - overlap

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')
#    print('DETECTIONS:::::',detections)
    return detections


def writeResultsToFile(detections, min_conf, path):

    print('WRITING RESULTS TO', path, '...', end=' ')
    rcnt = 0
    with open(path, 'w') as rfile:
        rfile.write('Start (s);End (s);Scientific name;Common name;Confidence\n')
        for d in detections:
            for entry in detections[d]:
                if entry[1] >= min_conf and ((entry[0] in INCLUDE_LIST or len(INCLUDE_LIST) == 0) and (entry[0] not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)):
                    rfile.write(d + ';' + entry[0].replace('_', ';') + ';' + str(entry[1]) + '\n')
                    rcnt += 1
    print('DONE! WROTE', rcnt, 'RESULTS.')
    return


def handle_client(conn, addr):
    global INCLUDE_LIST
    global EXCLUDE_LIST
    # print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False
            else:
                # print(f"[{addr}] {msg}")

                args = type('', (), {})()

                args.i = ''
                args.o = ''
                #args.birdweather_id = '99999'
                args.include_list = 'null'
                args.exclude_list = 'null'
                args.overlap = 0.0
                args.week = -1
                args.sensitivity = 1.25
                args.min_conf = 0.70
                args.lat = -1
                args.lon = -1

                for line in msg.split('||'):
                    inputvars = line.split('=')
                    if inputvars[0] == 'i':
                        args.i = inputvars[1]
                    elif inputvars[0] == 'o':
                        args.o = inputvars[1]
                    #elif inputvars[0] == 'birdweather_id':
                    #    args.birdweather_id = inputvars[1]
                    #elif inputvars[0] == 'include_list':
                    #    args.include_list = inputvars[1]
                    #elif inputvars[0] == 'exclude_list':
                    #    args.exclude_list = inputvars[1]
                    elif inputvars[0] == 'overlap':
                        args.overlap = float(inputvars[1])
                    elif inputvars[0] == 'week':
                        args.week = int(inputvars[1])
                    elif inputvars[0] == 'sensitivity':
                        args.sensitivity = float(inputvars[1])
                    elif inputvars[0] == 'min_conf':
                        args.min_conf = float(inputvars[1])
                    elif inputvars[0] == 'lat':
                        args.lat = float(inputvars[1])
                    elif inputvars[0] == 'lon':
                        args.lon = float(inputvars[1])
                

                min_conf = max(0.01, min(args.min_conf, 0.99))

                # Load custom species lists - INCLUDED and EXCLUDED
                """if not args.include_list == 'null':
                    INCLUDE_LIST = loadCustomSpeciesList(args.include_list)
                else:
                    INCLUDE_LIST = []

                if not args.exclude_list == 'null':
                    EXCLUDE_LIST = loadCustomSpeciesList(args.exclude_list)
                else:
                    EXCLUDE_LIST = []"""

                #birdweather_id = args.birdweather_id

                

                """
                This code takes a directory of audio files and runs a model to perform bat call detection and classification.
                It returns in a csv file the time of the detection, the species of the calls
                and the confidence level of the predicted species.
                """
                
                ####################################
                # Parameters to be set by the user #
                ####################################
                on_GPU = False   # True if tensorflow runs on GPU, False otherwise
                do_time_expansion = True  # set to True if audio is not already time expanded
                save_res = True    # True to save the results in a csv file and False otherwise
                chunk_size = 4.0    # The size of an audio chunk
                data_dir = 'data/' # path of the directory containing the audio files
                result_dir = 'results/'    # path to the directory where the results are saved
                model_dir = 'model/'  # path to the saved models
                model_name = "cnn2" # one of: 'batmen', 'cnn2',  'hybrid_cnn_svm',
                # 'hybrid_cnn_xgboost', 'hybrid_call_svm', 'hybrid_call_xgboost'

                # name of the result file
                classification_result_file = result_dir + 'classification_result.csv'
                if not os.path.isdir(result_dir):
                    os.makedirs(result_dir)

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

                # load model
                if model_name == "batmen":
                    date = "25_05_21_12_12_25_"
                    hnm_iter = "1" 
                    model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm_iter
                    network_classif = load_model(model_file_classif + '_model')
                elif model_name == "cnn2":
                    date = "25_05_21_15_09_28_" 
                    hnm_iter = "0"
                    model_file_detect = model_dir + date + "detect_" + model_name + "_hnm" + hnm_iter
                    network_detect = load_model(model_file_detect + '_model')
                    model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm_iter
                    network_classif = load_model(model_file_classif + '_model')
                elif model_name == "hybrid_cnn_svm":
                    date = "04_06_21_08_55_14_"
                    hnm_iter = "0"
                    model_file_features = model_dir + date + "features_" + model_name + "_hnm" + hnm_iter
                    network_features = load_model(model_file_features + '_model')
                    network_feat = Model(inputs=network_features.input, outputs=network_features.layers[-3].output)
                    model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm_iter
                    network_classif = joblib.load(model_file_classif + '_model.pkl')
                    scaler = joblib.load(model_file_classif + '_scaler.pkl' )
                elif model_name == "hybrid_cnn_xgboost":
                    date = "18_02_23_11_18_57_"
                    hnm_iter = "2"
                    model_file_features = model_dir + date + "features_" + model_name + "_hnm" + hnm_iter
                    network_features = load_model(model_file_features + '_model')
                    network_feat = Model(inputs=network_features.input, outputs=network_features.layers[-3].output)
                    model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm_iter
                    network_classif = joblib.load(model_file_classif + '_model.pkl')
                elif model_name == "hybrid_call_svm":
                    date = "26_05_21_08_40_42_"
                    hnm_iter = "0"
                    model_file_detect = model_dir + date + "detect_" + model_name + "_hnm" + hnm_iter
                    network_detect = load_model(model_file_detect + '_model')
                    model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm_iter
                    network_classif = joblib.load(model_file_classif + '_model.pkl')
                    scaler = joblib.load(model_file_classif + '_scaler.pkl' )
                elif model_name == "hybrid_call_xgboost":
                    date = "25_05_21_17_51_23_"
                    hnm_iter = "1"
                    model_file_detect = model_dir + date + "detect_" + model_name + "_hnm" + hnm_iter
                    network_detect = load_model(model_file_detect + '_model')
                    model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm_iter
                    network_classif = joblib.load(model_file_classif + '_model.pkl')
                
                # load params
                with open(model_file_classif + '_params.p') as f:
                    parameters = json.load(f)
                print("params=", parameters)

                # array with group name according to class number
                group_names = ['not call', 'Barbarg', 'Envsp', 'Myosp', 'Pip35','Pip50', 'Plesp', 'Rhisp']

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
                if model_name in ["hybrid_cnn_xgboost", "hybrid_call_xgboost"]: params.n_estimators = parameters["n_estimators"]
                params.load_features_from_file = False
                params.detect_time = 0
                params.classif_time = 0
                model_cls = clss.Classifier(params)
                if model_name in  ["batmen", "cnn2", "hybrid_cnn_svm", "hybrid_cnn_xgboost", "hybrid_call_svm", "hybrid_call_xgboost"]:
                    model_cls.model.network_classif = network_classif
                if model_name in ["cnn2", "hybrid_call_svm", "hybrid_call_xgboost"]:
                    model_cls.model.network_detect = network_detect
                if model_name in ["hybrid_cnn_svm", "hybrid_cnn_xgboost"]:
                    model_cls.model.network_features = network_features
                    model_cls.model.model_feat = network_feat
                if model_name in ["hybrid_cnn_svm", "hybrid_call_svm"]:
                    model_cls.model.scaler = scaler
                
                # load thresholds
                threshold_classes = np.load(model_file_classif + '_thresholds.npy')
                threshold_classes = threshold_classes / 100

                print("model name =", model_name)
                results = []
                # load audio file names and loop through them
                audio_files = glob.glob(data_dir + '*.wav')
                for file_cnt, file_name in enumerate(audio_files):
                    print("------------",file_name,"--------------")
                    file_name_root = file_name[len(data_dir):]

                    # read audio file - skip file if cannot read
                    read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_name,
                                            do_time_expansion, chunk_size, model_cls.params.window_size)
                    if read_fail:
                        continue
                    if file_dur>4:
                        # run classifier
                        tic = time.time()
                        call_time, call_prob, call_classes = run_classifier(model_cls, audio, file_dur, samp_rate, threshold_classes, chunk_size, do_time_expansion)
                        toc = time.time()
                        print("total time = ",toc-tic)
                        num_calls = len(call_time)
                        if num_calls>0:
                            call_classes = np.concatenate(np.array(call_classes)).ravel()
                            call_species = [group_names[i] for i in call_classes]
                            #print("call pos=",call_time)
                            #print("call species=", call_species)
                            #print("call proba=",call_prob)
                        print('  ' + str(num_calls) + ' calls found')

                        # save results
                        if save_res:
                            # save to AudioTagger format
                            op_file_name = result_dir + file_name_root[:-4] + '-sceneRect.csv'
                            wo.create_audio_tagger_op(file_name_root, op_file_name, call_time,
                                                    call_classes, call_prob,
                                                    samp_rate_orig, group_names)

                            # save as dictionary
                            if num_calls > 0:
                                res = {'filename':file_name_root, 'time':call_time,
                                    'prob':call_prob, 'pred_classes':call_species}
                                results.append(res)

                # save to large csv
                if save_res and (len(results) > 0):
                    print('\nsaving results to', classification_result_file)
                    wo.save_to_txt(classification_result_file, results, min_conf)
                else:
                    print('no detections to save')

    conn.close()


def start():
    # Load model
    global INTERPRETER, INCLUDE_LIST, EXCLUDE_LIST
    #INTERPRETER = loadModel()
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()
