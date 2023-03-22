from pathlib import Path
from tzlocal import get_localzone
import datetime
import sqlite3
import requests
import json
import time
import math
import numpy as np
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
MAXIMAL_ANSWER_LENGTH = 2048

userDir = os.path.expanduser('~')
DB_PATH = userDir + '/BirdNET-Pi/scripts/birds.db'

result_lock = threading.Lock()
perf_lock = threading.Lock()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    server.bind(ADDR)
except BaseException:
    print("Waiting on socket")
    time.sleep(5)


# Open most recent Configuration and grab DB_PWD as a python variable
with open(userDir +'/BirdNET-Pi/scripts/thisrun.txt', 'r') as f:
    this_run = f.readlines()
    audiofmt = "." + str(str(str([i for i in this_run if i.startswith('AUDIOFMT')]).split('=')[1]).split('\\')[0])
    priv_thresh = float("." + str(str(str([i for i in this_run if i.startswith('PRIVACY_THRESHOLD')]).split('=')[1]).split('\\')[0])) / 10

def recap(data, minimal_conf):
    #getting empty dictionnary (no key-values) ==> something wrong in code
    if not data:
        return "No result"
    else:
        payload = ""
        for i in range(len(data["prob"])):
            specie = data["pred_classes"][i]
            prob = data["prob"][i]
            if(prob>minimal_conf):
                new_payload = payload + str(specie) + " : " + str(prob) + "\n"
                if(len(new_payload.encode('utf-8'))>MAXIMAL_ANSWER_LENGTH):
                    return payload
                else:
                    payload = new_payload
def pre_loading_model():
    """
    This code takes a directory of audio files and runs a model to perform bat call detection and classification.
    It returns in a csv file the time of the detection, the species of the calls
    and the confidence level of the predicted species.
    """
    
    ####################################
    # Parameters to be set by the user #
    ####################################
    model_dir = userDir +'/BirdNET-Pi/model/'  # path to the saved models
    model_name = "cnn2" # one of: 'batmen', 'cnn2',  'hybrid_cnn_svm',
    # 'hybrid_cnn_xgboost', 'hybrid_call_svm', 'hybrid_call_xgboost'

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
    #print("params=", parameters)

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
    
    model_data = {}
    model_data["model_cls"] = model_cls
    model_data["group_names"] = group_names
    model_data["model_name"] = model_name
    model_data["model_file_classif"] = model_file_classif
    return model_data

def record_perf(data):

    path = userDir+'/BirdNET-Pi/perf_logs/'

    log_file_name = 'log.csv'
    if not os.path.exists(path+log_file_name) :
        with open(path+log_file_name, "w") as log_file:
            head_title = "timestamp_writing_perf,data_file,AI_used,nbr_detection,feat_dur, nms_dur,detection_dur,classication_dur,total_dur"
            log_file.write(head_title + '\n')
    with open(path + log_file_name, "a") as log_file:
        timestamp = str(time.strftime('%x-%X'))
        data_file = data["file"]
        ai_used = data["ai"]
        nbr_detect = data["nbr_detection"]
        feat_dur = data["feat_time"]
        nms_dur = data["nms_time"]
        detection_dur = data["detect_time"]
        classif_dur = data["classif_time"]
        tot_dur = data["tot_time"]
        payload = timestamp+","+data_file+","+ai_used+","+nbr_detect+","+feat_dur+","+nms_dur+","+detection_dur+","+classif_dur+","+tot_dur
        log_file.write(payload+"\n")

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
                #print(min_conf)
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
                
                
                # name of the result file
                classification_result_file = args.o    # path to the directory where the results are saved
                save_res = True    # True to save the results in a csv file and False otherwise
                on_GPU = False   # True if tensorflow runs on GPU, False otherwise
                chunk_size = 4.0    # The size of an audio chunk
                do_time_expansion = True  # set to True if audio is not already time expanded

                model_file_classif = MODEL["model_file_classif"]
                model_name = MODEL["model_name"]
                model_cls = MODEL["model_cls"]
                group_names = MODEL["group_names"]

                results = {}
                to_return = "nothing found"
                

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
                
                # load thresholds
                threshold_classes = np.load(model_file_classif + '_thresholds.npy')
                threshold_classes = threshold_classes / 100

                print("model name =", model_name)
                file_path = args.i
                file_name = [val for val in file_path.split("/")][-1]

                print("------------",file_name,"--------------")

                # read audio file - skip file if cannot read
                read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_path,
                                        do_time_expansion, chunk_size, model_cls.params.window_size)
                if read_fail:
                    continue
                if file_dur>4:
                    data={}
                    # run classifier
                    tic = time.time()
                    call_time, call_prob, call_classes, t = run_classifier(model_cls, audio, file_dur, samp_rate, threshold_classes, chunk_size, do_time_expansion)
                    toc = time.time()

                    # capturing duration
                    data["file"] =  file_name
                    data["ai"] = model_name
                    data["nbr_detection"] = str(max(0,len(call_classes)-1))
                    data["feat_time"] =  str(round(t["features"],3))
                    data["nms_time"] =  str(round(t["nms"],3))
                    data["detect_time"] =  str(round(t["detection"],3))
                    data["classif_time"] = str(round(t["classification"],3))
                    data["tot_time"] = str(round(toc-tic,3))
                    print("total time = ",toc-tic)

                    #need to avoid concurrence writing
                    perf_lock.acquire()     
                    print("WRITING PERF")               
                    record_perf(data)
                    perf_lock.release() 

                    num_calls = len(call_time)
                    if num_calls>0:
                        call_classes = np.concatenate(np.array(call_classes, dtype= object)).ravel()
                        call_species = [group_names[i] for i in call_classes]
                        #print("call pos=",call_time)
                        #print("call species=", call_species)
                        #print("call proba=",call_prob)
                    print('  ' + str(num_calls) + ' calls found')

                    # save results
                    if save_res:
                        #no use
                        """# save to AudioTagger format
                        op_file_name = file_name + '-sceneRect.csv'
                        wo.create_audio_tagger_op(file_name, op_file_name, call_time,
                                                call_classes, call_prob,
                                                samp_rate_orig, group_names)"""

                        # save as dictionary
                        if num_calls > 0:

                            results['filename'] = file_name
                            results['time'] = call_time
                            results['prob'] = call_prob
                            results['pred_classes'] = call_species
                            
                            # save to large csv
                            spliter_position = classification_result_file.rfind("/")
                            path_daily_result = classification_result_file[:spliter_position]
                            
                            print('\nsaving results to', path_daily_result)
                            print("min conf : "+str(min_conf))
                            
                            #need to avoid concurrence writing
                            result_lock.acquire()
                            print("wrinting result to file")
                            wo.save_to_txt(path_daily_result, results, min_conf)
                            result_lock.release()
                        else:
                            print('no detections to save')

                #answer to analyse.py
                to_return = str(recap(results, min_conf))
                print(to_return)
                conn.send(to_return.encode(FORMAT))
                session.close()

    conn.close()


def start():
    # Load model
    global MODEL, INCLUDE_LIST, EXCLUDE_LIST
    MODEL = pre_loading_model()
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()
