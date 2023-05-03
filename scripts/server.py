from batdetect2 import api
import librosa
import time
import numpy as np
import socket
import threading
import os

from utils.notifications import sendAppriseNotifications
from utils.parse_settings import config_to_settings

from bat_utils import write_op as wo

import numpy as np
import os
import time
#import tensorflow as tf



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


def record_perf(data):

    path = userDir+'/BirdNET-Pi/perf_logs/'
    log_file_name = 'ai_log.csv'

    if not os.path.exists(path+log_file_name) :

        with open(path+log_file_name, "w") as log_file:
            head_title = "timestamp_writing_perf,data_file,AI_used,nbr_detection,feat_dur, nms_dur,detection_dur,classication_dur,total_dur,nbr_thread,file_duration"
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
        thread = data["nbr_thread"]
        file_dur = data["file_dur"]

        payload = timestamp+","+data_file+","+ai_used+","+nbr_detect+","+feat_dur+","+nms_dur+","+detection_dur+","+classif_dur+","+tot_dur+","+thread+","+file_dur
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
                    elif inputvars[0] == 'nbr_thread':
                        args.nbr_thread = float(inputvars[1])

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
                confident_result = True
                nbr_detection = 0

                model_name = "Batdetect2"
                to_return = "nothing found"

                print("model name =", model_name)
                file_path = args.i
                file_name = [val for val in file_path.split("/")][-1]

                print("------------",file_name,"--------------")
                data={}
                # run classifier
                tic = time.time()
                results = api.process_file(file_path)["pred_dict"]
                toc = time.time()

                # capturing duration
                data["file"] =  results["id"]
                data["ai"] = model_name
                data["nbr_detection"] = str(len(results["annotation"]))
                data["feat_time"] =  str("unk")
                data["nms_time"] =  str("unk")
                data["detect_time"] =  str("unk")
                data["classif_time"] = str("unk")
                data["tot_time"] = str(round(toc-tic,3))
                data["nbr_thread"] = str(int(args.nbr_thread))
                data["file_dur"] = str(results["duration"])
                print("total time = ",toc-tic)

                #need to avoid concurrence writing
                perf_lock.acquire()     
                print("WRITING PERF")               
                record_perf(data)
                perf_lock.release() 

                # save results
                if save_res:
                    #no use
                    """# save to AudioTagger format
                    op_file_name = file_name + '-sceneRect.csv'
                    wo.create_audio_tagger_op(file_name, op_file_name, call_time,
                                            call_classes, call_prob,
                                            samp_rate_orig, group_names)"""

                    # save as csv file if enough confidence in result
                    if confident_result:

                        results['id'] = file_name
                        results['time'] = "unk"
                        results['prob'] = "unk"
                        #results['pred_classes'] = call_species
                        
                        # save to large csv
                        spliter_position = classification_result_file.rfind("/")
                        path_daily_result = classification_result_file[:spliter_position]
                        
                        print('\nsaving results to', path_daily_result)
                        #print("min conf : "+str(min_conf))
                        
                        #need to avoid concurrence writing
                        result_lock.acquire()
                        print("wrinting result to file")
                        nbr_detection = wo.save_batdetect2(path_daily_result, results, min_conf)
                        result_lock.release()

            #answer to analyse.py
                #to_return = '  ' + str(num_calls) + ' calls found'
                if nbr_detection>0:
                    to_return = f"{nbr_detection} calls found"
                else:
                    print('no detections to save')
                    os.system('rm '+file_path)
                    print('removing '+file_name)
                print(to_return)
                conn.send(to_return.encode(FORMAT))
                #session.close()

    conn.close()


def start():
    # Load model
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")
start()
