from scipy.io import wavfile
import numpy as np
import time as time
import os

def store_data_4debug(start_positions,  position_array, prob_array, classes_array):

    userDir = os.path.expanduser('~')
    path = userDir+'/BirdNET-Pi/perf_logs/'

    log_file_name = 'debug.csv'

    if not os.path.exists(path+log_file_name) :
        with open(path+log_file_name, "w") as log_file:
            head_title = "size_start_pos,shape_pos, prob_shape, shape_class\n"
            log_file.write(head_title + '\n')
    with open(path + log_file_name, "a") as log_file:
        payload = str(len(start_positions))+", "+str(position_array.shape)+", "+str(prob_array.shape)+", "+str(classes_array.shape)

        log_file.write(payload+"\n")

def read_audio(file_name, do_time_expansion, chunk_size, win_size):
    """
    Reads the audio file and apply time expansion if needed.

    Parameters
    -----------
    file_name : String
        Name of the audio file.
    do_time_expansion : bool
        True if time expansion need to be applied on the audio file and False otherwise.
    chunk_size : float
        Size of an audio chunk.
    win_size : float
        Size of a window.

    Returns
    --------
    read_fail : bool
        True if an error occurred while reading the file and False otherwised.
    audiopad : numpy array
        Audio samples padded with zeroes so the calls are not too close to the end of the file.
    file_dur : float
        Duration of the file.
    samp_rate : float
        Sampling rate of the file after a potential time expansion.
    samp_rate_orig : float
        Original sampling rate of the file.
    """

    # try to read in audio file
    try:
        samp_rate_orig, audio = wavfile.read(file_name)
    except:
        print('  Error reading file: ', file_name)
        return True, None, None, None, None

    # convert to mono if stereo
    if len(audio.shape) == 2:
        print('  Warning: stereo file. Just taking right channel.')
        audio = audio[:, 1]
    file_dur = audio.shape[0] / float(samp_rate_orig)

    # original model is trained on time expanded data
    samp_rate = samp_rate_orig
    if do_time_expansion:
        samp_rate = int(samp_rate_orig/10.0)
        file_dur *= 10

    # pad with zeros so we can go right to the end
    multiplier = np.ceil(file_dur/float(chunk_size-win_size))
    diff = multiplier*(chunk_size-win_size) - file_dur + win_size
    audio_pad = np.hstack((audio, np.zeros(int(diff*samp_rate))))

    read_fail = False
    return read_fail, audio_pad, file_dur, samp_rate, samp_rate_orig

def run_classifier(model, audio, file_path, file_dur, samp_rate, threshold_classes, chunk_size, do_time_expansion):
    """
    Uses the model to predict the time, class and confidence level of bat calls in the file.

    Parameters
    -----------
    model : Classifier
        Model used to detect and classify.
    audio : numpy array
        Audio samples of the file.
    file_path : String
        path of the wav file
    file_dur : float
        Duration of the file.
    samp_rate : float
        Sampling rate of the file.
    threshold_classes : numpy array
        Thresholds above which the confidence level needs to be to consider the prediction as a call.
        There is one threshold per class.
    chunk_size : float
        Size of an audio chunk.
    
    Returns
    --------
    call_time : numpy array
        Positions where calls are predicted in the file.
    call_prob : numpy array
        Confidence level of the predicted calls.
    call_class : list
        Classes of the predicted calls.
    """

    call_time = []
    call_prob = []
    call_class = []
    test_time = []

    # files can be long so we split each up into separate (overlapping) chunks
    st_positions = np.arange(0, file_dur, chunk_size-model.params.window_size)
    for chunk_id, st_position in enumerate(st_positions):

        # take a chunk of the audio
        st_pos = int(st_position*samp_rate)
        en_pos = int(st_pos + chunk_size*samp_rate)
        audio_chunk = audio[st_pos:en_pos]

        # make predictions
        tic = time.time()
        pos, prob, classes = model.test_single(file_path, audio_chunk, samp_rate)
        toc = time.time()
        test_time.append(round(toc-tic, 3))
        store_data_4debug(st_position, pos, prob, classes)
        if pos.shape[0] > 0:
            prob = prob[:, 0]

            # remove predictions near the end (if not last chunk) and ones that are
            # below the detection threshold
            if chunk_id == (len(st_positions)-1):
                inds = (prob >= threshold_classes[classes])
            else:
                inds = (prob >= threshold_classes[classes]) & (pos < (chunk_size-(model.params.window_size/2.0)))

            # keep valid detections and convert detection time back into global time
            if pos.shape[0] > 0:
                call_time.append(pos[inds] + st_position)
                call_prob.append(prob[inds])
                call_class.append(classes[inds])

    if len(call_time) > 0:
        call_time = np.hstack(call_time)
        call_prob = np.hstack(call_prob)

        # undo the effects of times expansion
        if do_time_expansion:
            call_time /= 10.0
    
    print('chunk time', np.mean(test_time), '(secs)')
    print('nb chunks', len(st_positions))
    print('features computation time', model.params.features_computation_time, '(secs)')
    print('nms computation time', model.params.nms_computation_time, '(secs)')
    print('detect time total', model.params.detect_time, '(secs)')
    print('classif time total', model.params.classif_time, '(secs)')
    tps = {}
    tps["features"] = model.params.features_computation_time
    tps["nms"] = model.params.nms_computation_time
    tps["detection"] = model.params.detect_time
    tps["classification"] = model.params.classif_time
    return call_time, call_prob, call_class, tps