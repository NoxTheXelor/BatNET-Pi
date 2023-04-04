import speed_bat_utils.cls_hybrid_cnn as cls_hybrid_cnn


class Classifier:

    def __init__(self, params_):
        """
        Creates a new classifier.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """
        
        self.params = params_
        self.model = cls_hybrid_cnn.NeuralNet(self.params)

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
        self.model.save_features(goal, files)
    
    def test_single(self, full_path, audio_samples, sampling_rate):
        """
        Makes a prediction on the position, probability and class of the calls present in the raw audio samples.

        Parameters
        -----------
        full_path : String
            path of the wav file
        audio_samples : numpy array
            Data read from a wav file.
        sampling_rate : int
            Sample rate of a wav file.
        
        Returns
        --------
        nms_pos : numpy array
            Predicted positions of calls.
        nms_prob : numpy array
            Confidence level of each prediction.
        pred_classes : numpy array
            Predicted class of each prediction.
        """
        duration = audio_samples.shape[0]/float(sampling_rate)
        nms_pos, nms_prob, pred_classes, nb_windows = self.model.test("classification", full_path, file_duration=duration, audio_samples=audio_samples, sampling_rate=sampling_rate) # modif: renvoit aussi matches=classes
        return nms_pos, nms_prob, pred_classes, nb_windows

    def test_batch(self, goal, full_path, file, durations):
        """
        Makes a prediction on the position, probability and class of the calls present in the list of audio files.

        Parameters
        -----------
        goal : String
            Indicates whether the files need to be tested for detection or classification.
            Can be either "detection" or "classification".
        full_path : String
            path of the wav file
        file : String
            Name of the wav file used to test the model.
        durations : numpy array
            Durations of the wav files used to test the model.

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
        nms_pos = None
        nms_prob = None
        pred_classes = None
        nb_windows = None
        #file_name = "20200806_230000T"
        nms_pos, nms_prob, pred_classes, nb_windows = self.model.test(goal, full_path,file_name=file,
                                                                          file_duration=durations) #110.0
        return nms_pos, nms_prob, pred_classes, nb_windows
