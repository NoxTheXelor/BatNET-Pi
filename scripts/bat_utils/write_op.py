import pandas as pd
import numpy as np
import datetime as dt
import os

def save_to_txt(op_file, results, min_conf):
    """
    Takes a list of dictionaries of results and saves them to file.

    Parameters
    -----------
    op_file : String
        Path to the file in which the results will be saved.
    results : list
        Contains dictionaries with each the four following fields: filename, time, prob, pred_classes.
    min_conf : float
        minimum threshold of confidence in species prediction. Below this threshold, the data is not written in the file.
    """
    if not os.path.exists(op_file+'/daily_result.csv'):
        with open(op_file+'/daily_result.csv', 'w') as file:
            head_str = 'file_name,predicted_time,predicted_species,predicted_prob'
            file.write(head_str + '\n')
   
    with open(op_file+'/daily_result.csv', 'a') as filling_file:
            for jj in range(len(results['prob'])):
                row_str = results['filename'] + ','
                tm = round(results['time'][jj],3)
                sp = results['pred_classes'][jj]
                pr = round(results['prob'][jj],3)

                if(pr>=min_conf):

                    row_str += str(tm) + ',' +str(sp) + ',' + str(pr)
                    filling_file.write(row_str + '\n')

def save_batdetect2(op_file, results, min_conf):
    """
    Takes a list of dictionaries of results and saves them to file.

    Parameters
    -----------
    op_file : String
        Path to the file in which the results will be saved.
    results : list
        Contains dictionaries with each the four following fields: filename, time, prob, pred_classes.
    min_conf : float
        minimum threshold of confidence in species prediction. Below this threshold, the data is not written in the file.
    """
    if not os.path.exists(op_file+'/daily_result.csv'):
        with open(op_file+'/daily_result.csv', 'w') as file:
            head_str = 'file_name,start_call,end_call,confidence_detection,predicted_species,confidence_pred'
            file.write(head_str + '\n')
   
    with open(op_file+'/daily_result.csv', 'a') as filling_file:
            
            data = results['annotation']
            counter = 0
            for jj in range(len(data)):
                
                row_str = results['id'] + ','
                start_time = data[jj]['start_time']
                end_time = data[jj]['end_time']
                detection_prediction = data[jj]['class_prob']
                specie = data[jj]["class"]
                specie_prediction = data[jj]['class_prob']
                #print("getting data")

                if specie_prediction>=min_conf and detection_prediction>=min_conf:
                    #print("writing data")
                    counter +=1
                    row_str += str(start_time) + ',' +str(end_time) + ',' +str(detection_prediction) + ',' +str(specie) + ',' + str(specie_prediction)
                    filling_file.write(row_str + '\n')
            return counter


def create_audio_tagger_op(ip_file_name, op_file_name, st_times,
                           class_pred, class_prob, samp_rate, class_names):
    """
    Saves the detections in an audiotagger friendly format.

    Parameters
    -----------
    ip_file_name : String
        Name of the wav file.
    op_file_name : String
        Name of the csv in which the results are saved.
    st_times : numpy array
        Starting times of the predicted calls.
    class_pred : list
        Classes of the predicted calls.
    class_prob : numpy array
        Confidence levels of the predicted calls.
    samp_rate : float
        Sampling rate of the file.
    class_names : list
        Maps the class number to the class name.
    
    """

    col_names = ['Filename', 'Label', 'LabelTimeStamp', 'Spec_NStep',
                 'Spec_NWin', 'Spec_x1', 'Spec_y1', 'Spec_x2', 'Spec_y2',
                 'LabelStartTime_Seconds', 'LabelEndTime_Seconds',
                 'LabelArea_DataPoints', 'ClassifierConfidence']

    nstep = 0.001
    nwin = 0.003
    call_width = 0.001  # code does not output call width so just put in dummy value (batdetective)
    y_max = (samp_rate*nwin)/2.0
    num_calls = len(st_times)


    if num_calls > 0:
        da_at = pd.DataFrame(index=np.arange(0, num_calls), columns=col_names)
        da_at['Spec_NStep'] = nstep
        da_at['Spec_NWin'] = nwin
        da_at['Label'] = 'bat'
        da_at['LabelTimeStamp'] = dt.datetime.now().isoformat()
        da_at['Spec_y1'] = 0
        da_at['Spec_y2'] = y_max
        da_at['Filename'] = ip_file_name

        for ii in np.arange(0, num_calls):
            st_time = st_times[ii]
            da_at.loc[ii, 'LabelStartTime_Seconds'] = st_time
            da_at.loc[ii, 'LabelEndTime_Seconds'] = st_time + call_width
            da_at.loc[ii, 'Label'] = class_names[class_pred[ii]]
            da_at.loc[ii, 'Spec_x1'] = st_time/nstep
            da_at.loc[ii, 'Spec_x2'] = (st_time + call_width)/nstep
            da_at.loc[ii, 'ClassifierConfidence'] = round(class_prob[ii], 3)

        # save to disk
        da_at.to_csv(op_file_name, index=False)
    else:
        "No result to save"
