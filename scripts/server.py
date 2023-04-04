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
import librosa

from utils.notifications import sendAppriseNotifications
from utils.parse_settings import config_to_settings

from speed_bat_utils import classifier as clss
from speed_bat_utils import write_op as wo
from speed_bat_utils.data_set_params import DataSetParams
from speed_bat_utils.audio import read_audio, run_classifier
#from tensorflow.keras.models import Model, load_model
#from speed_bat_utils.run_classifier import read_audio, run_classifier

from scipy.io import wavfile
import numpy as np
import os
import glob
import time
#import tensorflow as tf
import json
#from tensorflow.keras.models import Model, load_model
import joblib

from tflite_support.metadata_writers import writer_utils
from speed_bat_utils.larq_compute_engine.tflite.python import interpreter
import xgboost as xgb


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

def pre_loading_model(path):
    """
    model_dir : str - name of the used model. Can be either 
        model_raspberry/ 
        raspberry_model_V2/ ==> xgboost with 500 estimators
        model_float/         ==> float model
    """
    load_features_from_file = False
    model_name = "hybrid_cnn_xgboost"  # can be one of: 'batmen', 'cnn2',  'hybrid_cnn_svm', 'hybrid_cnn_xgboost', 'hybrid_call_svm', 'hybrid_call_xgboost'
    #model_dir = 'model_raspberry/' # Binary model
    model_dir = path+'speed_bat_utils/raspberry_model_V2/' # Binary model with another XGBoost model using 500 estimators
    #model_dir = "model_float/" # Float model

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
    model_cls = clss.Classifier(params)

    model_cls.model.network_classif = network_classif
    model_cls.model.network_features = None
    model_cls.model.model_feat = network_feat

    return model_cls

def record_perf(data):

    path = userDir+'/BirdNET-Pi/perf_logs/'

    log_file_name = 'log.csv'
    if not os.path.exists(path+log_file_name) :
        with open(path+log_file_name, "w") as log_file:
            head_title = "timestamp_writing_perf,data_file,AI_used,nbr_detection,total_dur\n"
            log_file.write(head_title + '\n')
    with open(path + log_file_name, "a") as log_file:
        timestamp = str(time.strftime('%x-%X'))
        data_file = data["file"]
        ai_used = data["ai"]
        nbr_detect = data["nbr_detection"]
        tot_dur = data["tot_time"]
        payload = timestamp+","+data_file+","+ai_used+","+nbr_detect+","+tot_dur
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
                # array with group name according to class number
                results = []
                save_res = True
                do_time_expansion = True  # set to True if audio is not already time expanded
                chunk_size = 4.0    # The size of an audio chunk
                group_names = ['not call', 'Barbarg', 'Envsp', 'Myosp', 'Pip35', ' Pip50', 'Plesp', 'Rhisp']
                classification_result_file = args.o
                path_file = args.i
                # read audio file - skip file if cannot read
                read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_name,
                                        do_time_expansion, chunk_size, MODEL.params.window_size)
                if read_fail:
                    continue
                if file_dur>4:
                    file_name = [val for val in path_file.split("/")][-1]
                    tic = time.time()
                    #call_time, call_prob, call_classes, nb_window = MODEL.test_batch("classification", path_file,file_name,file_dur)
                    call_time, call_prob, call_classes, nb_window = run_classifier(MODEL, audio, file_dur, samp_rate, threshold_classes, chunk_size)
                    toc = time.time()
                    data = {}
                    data["file"] =  file_name
                    data["ai"] = "XgBoost"
                    data["nbr_detection"] = str(max(0,len(call_classes)-1))
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
                            res = {'filename':file_name, 'time':call_time,
                                'prob':call_prob, 'pred_classes':call_species}
                            results.append(res)

                spliter_position = classification_result_file.rfind("/")
                path_daily_result = classification_result_file[:spliter_position]
                # save to large csv
                if save_res and (len(results) > 0):
                    print('\nsaving results to', path_daily_result)
                    print("min conf : "+str(min_conf))
                    #need to avoid concurrence writing
                    result_lock.acquire()
                    print("wrinting result to file")
                    wo.save_to_txt(path_daily_result, results, min_conf)
                    result_lock.release()
                else:
                    print('no detections to save')
                    os.system('rm '+ path_file)
                    print(file_name+' removed')

    conn.close()


def start():
    # Load model
    global MODEL, INCLUDE_LIST, EXCLUDE_LIST
    MODEL = pre_loading_model(userDir+'/BirdNET-Pi/scripts/')
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()
