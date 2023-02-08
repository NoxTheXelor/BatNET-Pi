import numpy as np
from hyperopt import hp, tpe, fmin, space_eval, Trials
from hyperopt.pyll.base import scope
import pickle
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import gc

from bat_utils.data_set_params import DataSetParams

def tune_svm(params, feat_train, labels, trials_filename):
    """
    Tunes the SVM with hyperopt.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    feat_train : ndarray
        Array containing the spectrogram features for each window of the audio file.
    labels : numpy array
        Class label (0-7) for each training position.
    trials_filename : String
        Name of the file where the previous iterations of hyperopt are saved.
    """

    print("\n tune svm")
    space_svm = {   'C':hp.choice('C', [0.1, 1, 10, 100, 1000]),
                    'kernel':hp.choice('kernel',['linear','rbf','poly','sigmoid']),
                    'degree': scope.int(hp.quniform('degree',1,15,1)),
                    'gamma_svm': hp.choice('gamma_svm', ['auto', 0.1, 1, 10, 100]),
                    'class_weight':hp.choice('class_weight', ["balanced", None]),
                    'max_iter': hp.choice('max_iter', [100,500,1000,1500,2000,2500]),
                    'model': "svm",
                    'feat_train': feat_train,
                    'labels': labels
                }

    # load the saved trials
    try:
        trials = pickle.load(open(trials_filename+".hyperopt", "rb"))
        max_trials = len(trials.trials) + 1
    # create a new trials
    except:
        max_trials = 1
        trials = Trials()
    
    # optimise the objective function with the defined set of SVM parameters
    best_space_indices = fmin(obj_func_svm_xgb, space_svm, trials=trials, algo=tpe.suggest, max_evals=max_trials)
    best_space = space_eval(space_svm, best_space_indices)
    best_space = {k: best_space[k] for k in best_space.keys() - {'feat_train', 'labels'}}
    with open(trials_filename + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)

    params.C = best_space['C']
    params.kernel = best_space['kernel']
    params.degree = best_space['degree']
    params.gamma_svm = best_space['gamma_svm']
    params.class_weight = best_space['class_weight']
    params.max_iter = best_space['max_iter']



def tune_xgb(params, feat_train, labels , trials_filename):
    """
    Tunes the XGBoost with hyperopt.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    feat_train : ndarray
        Array containing the spectrogram features for each window of the audio file.
    labels : numpy array
        Class label (0-7) for each training position.
    trials_filename : String
        Name of the file where the previous iterations of hyperopt are saved.
    """

    print("\n tune xgboost")
    space_xgb = {   'eta': hp.choice('eta', np.logspace(-4, -0.522879, num=20)),
                    'min_child_weight': hp.choice('min_child_weight', [1,2,3]),
                    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
                    'n_estimators': hp.choice('n_estimators', [100, 500, 1000, 2000]),
                    'gamma_xgb': hp.choice('gamma_xgb', [0, 0.0001, 0.005, 0.001, 0.005, 0.01]),
                    'subsample': hp.choice('subsample', [0.7, 0.8, 0.9, 1]),
                    'scale_pos_weight': hp.choice('scale_pos_weight', [0, 0.25, 0.5, 1, 1.5]),
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'model': "xgboost",
                    'feat_train': feat_train,
                    'labels': labels
                }

    
    # load the saved trials
    try:
        trials = pickle.load(open(trials_filename+".hyperopt", "rb"))
        max_trials = len(trials.trials) + 1
    # create a new trials
    except:
        max_trials = 1
        trials = Trials()

    # optimise the objective function with the defined set of XGBoost parameters
    best_space_indices = fmin(obj_func_svm_xgb, space_xgb, trials=trials, algo=tpe.suggest, max_evals=max_trials)
    best_space = space_eval(space_xgb, best_space_indices)
    best_space = {k: best_space[k] for k in best_space.keys() - {'feat_train', 'labels'}}
    with open(trials_filename + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)

    params.eta = best_space['eta']
    params.min_child_weight = best_space['min_child_weight']
    params.max_depth = best_space['max_depth']
    params.n_estimators = best_space['n_estimators']
    params.gamma_xgb = best_space['gamma_xgb']
    params.subsample = best_space['subsample']
    params.scale_pos_weight = best_space['scale_pos_weight']


def obj_func_svm_xgb(args):
    """
    Fits and returns the best loss of an SVM or XGBoost model with given parameters.

    Parameters
    -----------
    args : dict
        Dictionnary of all the parameters needed to fit an SVM or XGBoost.

    Returns
    --------
    loss : float
        minimum value of the loss during training of the SVM or XGBoost.
    """

    # split dataset into training and validation set
    train_feat, val_feat, train_labels, val_labels = train_test_split(args['feat_train'], args['labels'],
                                                        test_size=0.1, random_state=1, stratify=args['labels'])
    
    # compute sample weights
    class_weights = list(class_weight.compute_class_weight('balanced', np.unique(train_labels),train_labels))
    class_w = np.ones(train_labels.shape[0], dtype = 'float')
    for i, val in enumerate(train_labels):
        class_w[i] = class_weights[val]
    class_w_valid = np.ones(val_labels.shape[0], dtype = 'float')
    for i, val in enumerate(val_labels):
        class_w_valid[i] = class_weights[val]

    # Fit the SVM and compute the loss
    if args['model']=='svm':
        params_svm = DataSetParams()
        params_svm.C = args['C']
        params_svm.kernel = args['kernel']
        params_svm.degree = args['degree']
        params_svm.gamma_svm = args['gamma_svm']
        params_svm.class_weight = args['class_weight']
        params_svm.max_iter = args['max_iter']
        print_params_svm(params_svm)
        clf = SVC(kernel=params_svm.kernel, C=params_svm.C, degree=params_svm.degree,
                    gamma=params_svm.gamma_svm, class_weight=params_svm.class_weight, probability=True,
                    verbose=False, max_iter=params_svm.max_iter)
        clf.fit(train_feat, train_labels)
        y_pred_train = clf.predict_proba(val_feat)
        loss = log_loss(val_labels, y_pred_train)

    # Fit XGBoost and compute minimum loss
    elif args['model']=='xgboost':
        params_xgb = DataSetParams()
        params_xgb.eta = args['eta']
        params_xgb.min_child_weight = args['min_child_weight']
        params_xgb.max_depth = args['max_depth']
        params_xgb.n_estimators = args['n_estimators']
        params_xgb.gamma_xgb = args['gamma_xgb']
        params_xgb.subsample = args['subsample']
        params_xgb.scale_pos_weight = args['scale_pos_weight']
        print_params_xgb(params_xgb)
        clf = xgb.XGBClassifier(eta=params_xgb.eta, min_child_weight=params_xgb.min_child_weight, max_depth=params_xgb.max_depth,
                                n_estimators=params_xgb.n_estimators, gamma=params_xgb.gamma_xgb, subsample=params_xgb.subsample,
                                scale_pos_weight=params_xgb.scale_pos_weight, objective="multi:softprob",
                                tree_method='gpu_hist', predictor='gpu_predictor')
        clf.fit(train_feat, train_labels, eval_metric=['mlogloss'], sample_weight=class_w,
                early_stopping_rounds=params_xgb.n_estimators//10, eval_set=[(val_feat,val_labels)], verbose=False)
        history = clf.evals_result()
        loss = np.min(history['validation_0']['mlogloss'])
        clf._Booster.__del__()
    
    gc.collect()
    return loss


def print_params_svm(params):
    """
    Print the parameters of the SVM.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    """

    dic = {}
    dic["C"] = params.C
    dic["kernel"] = params.kernel
    dic["degree"] = params.degree 
    dic["gamma_svm"] = params.gamma_svm 
    dic["class_weight"] = params.class_weight
    dic["max_iter"] = params.max_iter
    print("params used for SVM =",dic)

def print_params_xgb(params):
    """
    Print the parameters of XGBoost.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    """

    dic = {}
    dic["eta"] = params.eta
    dic['min_child_weight'] = params.min_child_weight
    dic["max_depth"] = params.max_depth
    dic["n_estimators"] = params.n_estimators
    dic["gamma_xgb"] = params.gamma_xgb
    dic["subsample"] = params.subsample
    dic["scale_pos_weight"] = params.scale_pos_weight
    print("params used for XGBoost =", dic)