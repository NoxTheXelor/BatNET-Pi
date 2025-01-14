"""
Evaluates trained model on test set and generates plots.
"""

import argparse
import copy
import json
import os

import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from batdetect2.detector import parameters
import batdetect2.train.evaluate as evl
import batdetect2.train.train_utils as tu
import batdetect2.utils.detector_utils as du
import batdetect2.utils.plot_utils as pu


def get_blank_annotation(ip_str):

    res = {}
    res["class_name"] = ""
    res["duration"] = -1
    res["id"] = ""  # fileName
    res["issues"] = False
    res["notes"] = ip_str
    res["time_exp"] = 1
    res["annotated"] = False
    res["annotation"] = []

    ann = {}
    ann["class"] = ""
    ann["event"] = "Echolocation"
    ann["individual"] = -1
    ann["start_time"] = -1
    ann["end_time"] = -1
    ann["low_freq"] = -1
    ann["high_freq"] = -1
    ann["confidence"] = -1

    return copy.deepcopy(res), copy.deepcopy(ann)


def create_genus_mapping(gt_test, preds, class_names):
    # rolls the per class predictions and ground truth back up to genus level
    class_names_genus, cls_to_genus = np.unique(
        [cc.split(" ")[0] for cc in class_names], return_inverse=True
    )
    genus_to_cls_map = [
        np.where(np.array(cls_to_genus) == cc)[0]
        for cc in range(len(class_names_genus))
    ]

    gt_test_g = []
    for gg in gt_test:
        gg_g = copy.deepcopy(gg)
        inds = np.where(gg_g["class_ids"] != -1)[0]
        gg_g["class_ids"][inds] = cls_to_genus[gg_g["class_ids"][inds]]
        gt_test_g.append(gg_g)

    # note, will have entries geater than one as we are summing across the respective classes
    preds_g = []
    for pp in preds:
        pp_g = copy.deepcopy(pp)
        pp_g["class_probs"] = np.zeros(
            (len(class_names_genus), pp_g["class_probs"].shape[1]),
            dtype=np.float32,
        )
        for cc, inds in enumerate(genus_to_cls_map):
            pp_g["class_probs"][cc, :] = pp["class_probs"][inds, :].sum(0)
        preds_g.append(pp_g)

    return class_names_genus, preds_g, gt_test_g


def load_tadarida_pred(ip_dir, dataset, file_of_interest):

    res, ann = get_blank_annotation("Generated by Tadarida")

    # create the annotations in the correct format
    da_c = pd.read_csv(
        ip_dir
        + dataset
        + "/"
        + file_of_interest.replace(".wav", ".ta").replace(".WAV", ".ta"),
        sep="\t",
    )

    res_c = copy.deepcopy(res)
    res_c["id"] = file_of_interest
    res_c["dataset"] = dataset
    res_c["feats"] = da_c.iloc[:, 6:].values.astype(np.float32)

    if da_c.shape[0] > 0:
        res_c["class_name"] = ""
        res_c["class_prob"] = 0.0

    for aa in range(da_c.shape[0]):
        ann_c = copy.deepcopy(ann)
        ann_c["class"] = "Not Bat"  # will assign to class later
        ann_c["start_time"] = np.round(da_c.iloc[aa]["StTime"] / 1000.0, 5)
        ann_c["end_time"] = np.round(
            (da_c.iloc[aa]["StTime"] + da_c.iloc[aa]["Dur"]) / 1000.0, 5
        )
        ann_c["low_freq"] = np.round(da_c.iloc[aa]["Fmin"] * 1000.0, 2)
        ann_c["high_freq"] = np.round(da_c.iloc[aa]["Fmax"] * 1000.0, 2)
        ann_c["det_prob"] = 0.0
        res_c["annotation"].append(ann_c)

    return res_c


def load_sonobat_meta(
    ip_dir,
    datasets,
    region_classifier,
    class_names,
    only_accepted_species=True,
):

    sp_dict = {}
    for ss in class_names:
        sp_key = ss.split(" ")[0][:3] + ss.split(" ")[1][:3]
        sp_dict[sp_key] = ss

    sp_dict["x"] = ""  # not bat
    sp_dict["Bat"] = "Bat"

    sonobat_meta = {}
    for tt in datasets:
        dataset = tt["dataset_name"]
        sb_ip_dir = ip_dir + dataset + "/" + region_classifier + "/"

        # load the call level predictions
        ip_file_p = sb_ip_dir + dataset + "_Parameters_v4.5.0.txt"
        # ip_file_p = sb_ip_dir + 'audio_SonoBatch_v30.0 beta.txt'
        da = pd.read_csv(ip_file_p, sep="\t")

        # load the file level predictions
        ip_file_b = sb_ip_dir + dataset + "_SonoBatch_v4.5.0.txt"
        # ip_file_b = sb_ip_dir + 'audio_CumulativeParameters_v30.0 beta.txt'

        with open(ip_file_b) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        del lines[0]

        file_res = {}
        for ll in lines:
            # note this does not seem to parse the file very well
            ll_data = ll.split("\t")

            # there are sometimes many different species names per file
            if only_accepted_species:
                # only choosing "SppAccp"
                ind = 4
            else:
                # choosing ""~Spp" if "SppAccp" does not exist
                if ll_data[4] != "x":
                    ind = 4  # choosing "SppAccp", along with "Prob" here
                else:
                    ind = 8  # choosing "~Spp", along with "~Prob" here

            sp_name_1 = sp_dict[ll_data[ind]]
            prob_1 = ll_data[ind + 1]
            if prob_1 == "x":
                prob_1 = 0.0
            file_res[ll_data[1]] = {
                "id": ll_data[1],
                "species_1": sp_name_1,
                "prob_1": prob_1,
            }

        sonobat_meta[dataset] = {}
        sonobat_meta[dataset]["file_res"] = file_res
        sonobat_meta[dataset]["call_info"] = da

    return sonobat_meta


def load_sonobat_preds(dataset, id, sb_meta, set_class_name=None):

    # create the annotations in the correct format
    res, ann = get_blank_annotation("Generated by Sonobat")
    res_c = copy.deepcopy(res)
    res_c["id"] = id
    res_c["dataset"] = dataset

    da = sb_meta[dataset]["call_info"]
    da_c = da[da["Filename"] == id]

    file_res = sb_meta[dataset]["file_res"]
    res_c["feats"] = np.zeros((0, 0))

    if da_c.shape[0] > 0:
        res_c["class_name"] = file_res[id]["species_1"]
        res_c["class_prob"] = file_res[id]["prob_1"]
        res_c["feats"] = da_c.iloc[:, 3:105].values.astype(np.float32)

        for aa in range(da_c.shape[0]):
            ann_c = copy.deepcopy(ann)
            if set_class_name is None:
                ann_c["class"] = file_res[id]["species_1"]
            else:
                ann_c["class"] = set_class_name
            ann_c["start_time"] = np.round(
                da_c.iloc[aa]["TimeInFile"] / 1000.0, 5
            )
            ann_c["end_time"] = np.round(
                ann_c["start_time"] + da_c.iloc[aa]["CallDuration"] / 1000.0, 5
            )
            ann_c["low_freq"] = np.round(da_c.iloc[aa]["LowFreq"] * 1000.0, 2)
            ann_c["high_freq"] = np.round(da_c.iloc[aa]["HiFreq"] * 1000.0, 2)
            ann_c["det_prob"] = np.round(da_c.iloc[aa]["Quality"], 3)
            res_c["annotation"].append(ann_c)

    return res_c


def bb_overlap(bb_g_in, bb_p_in):

    freq_scale = 10000000.0  # ensure that both axis are roughly the same range
    bb_g = [
        bb_g_in["start_time"],
        bb_g_in["low_freq"] / freq_scale,
        bb_g_in["end_time"],
        bb_g_in["high_freq"] / freq_scale,
    ]
    bb_p = [
        bb_p_in["start_time"],
        bb_p_in["low_freq"] / freq_scale,
        bb_p_in["end_time"],
        bb_p_in["high_freq"] / freq_scale,
    ]

    xA = max(bb_g[0], bb_p[0])
    yA = max(bb_g[1], bb_p[1])
    xB = min(bb_g[2], bb_p[2])
    yB = min(bb_g[3], bb_p[3])

    # compute the area of intersection rectangle
    inter_area = abs(max((xB - xA, 0.0)) * max((yB - yA), 0.0))

    if inter_area == 0:
        iou = 0.0

    else:
        # compute the area of both
        bb_area_g = abs((bb_g[2] - bb_g[0]) * (bb_g[3] - bb_g[1]))
        bb_area_p = abs((bb_p[2] - bb_p[0]) * (bb_p[3] - bb_p[1]))

        iou = inter_area / float(bb_area_g + bb_area_p - inter_area)

    return iou


def assign_to_gt(gt, pred, iou_thresh):
    # this will edit pred in place

    num_preds = len(pred["annotation"])
    num_gts = len(gt["annotation"])
    if num_preds > 0 and num_gts > 0:
        iou_m = np.zeros((num_preds, num_gts))
        for ii in range(num_preds):
            for jj in range(num_gts):
                iou_m[ii, jj] = bb_overlap(
                    gt["annotation"][jj], pred["annotation"][ii]
                )

        # greedily assign detections to ground truths
        # needs to be greater than some threshold and we cannot assign GT
        # to more than one detection
        # TODO could try to do an optimal assignment
        for jj in range(num_gts):
            max_iou = np.argmax(iou_m[:, jj])
            if iou_m[max_iou, jj] > iou_thresh:
                pred["annotation"][max_iou]["class"] = gt["annotation"][jj][
                    "class"
                ]
                iou_m[max_iou, :] = -1.0

    return pred


def parse_data(data, class_names, non_event_classes, is_pred=False):
    class_names_all = class_names + non_event_classes

    data["class_names"] = np.array([aa["class"] for aa in data["annotation"]])
    data["start_times"] = np.array(
        [aa["start_time"] for aa in data["annotation"]]
    )
    data["end_times"] = np.array([aa["end_time"] for aa in data["annotation"]])
    data["high_freqs"] = np.array(
        [float(aa["high_freq"]) for aa in data["annotation"]]
    )
    data["low_freqs"] = np.array(
        [float(aa["low_freq"]) for aa in data["annotation"]]
    )

    if is_pred:
        # when loading predictions
        data["det_probs"] = np.array(
            [float(aa["det_prob"]) for aa in data["annotation"]]
        )
        data["class_probs"] = np.zeros(
            (len(class_names) + 1, len(data["annotation"]))
        )
        data["class_ids"] = np.array(
            [class_names_all.index(aa["class"]) for aa in data["annotation"]]
        ).astype(np.int32)
    else:
        # when loading ground truth
        # if the class label is not in the set of interest then set to -1
        labels = []
        for aa in data["annotation"]:
            if aa["class"] in class_names:
                labels.append(class_names_all.index(aa["class"]))
            else:
                labels.append(-1)
        data["class_ids"] = np.array(labels).astype(np.int32)

    return data


def load_gt_data(datasets, events_of_interest, class_names, classes_to_ignore):
    gt_data = []
    for dd in datasets:
        print("\n" + dd["dataset_name"])
        gt_dataset = tu.load_set_of_anns(
            [dd], events_of_interest=events_of_interest, verbose=True
        )
        gt_dataset = [
            parse_data(gg, class_names, classes_to_ignore, False)
            for gg in gt_dataset
        ]

        for gt in gt_dataset:
            gt["dataset_name"] = dd["dataset_name"]

        gt_data.extend(gt_dataset)

    return gt_data


def train_rf_model(x_train, y_train, num_classes, seed=2001):
    # TODO search for the best hyper parameters on val set
    # Currently only training on the species and 'not bat' - exclude 'generic_class' which is last
    # alternative would be to first have a "bat" vs "not bat" classifier, and then a species classifier?

    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    inds = np.where(y_train < num_classes)[0]
    x_train = x_train[inds, :]
    y_train = y_train[inds]
    un_train_class = np.unique(y_train)

    clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    tr_acc = (y_pred == y_train).mean()
    # print('Train acc', round(tr_acc*100, 2))
    return clf, un_train_class


def eval_rf_model(clf, pred, un_train_class, num_classes):
    # stores the prediction in place
    if pred["feats"].shape[0] > 0:
        pred["class_probs"] = np.zeros((num_classes, pred["feats"].shape[0]))
        pred["class_probs"][un_train_class, :] = clf.predict_proba(
            pred["feats"]
        ).T
        pred["det_probs"] = pred["class_probs"][:-1, :].sum(0)
    else:
        pred["class_probs"] = np.zeros((num_classes, 0))
        pred["det_probs"] = np.zeros(0)
    return pred


def save_summary_to_json(op_dir, mod_name, results):
    op = {}
    op["avg_prec"] = round(results["avg_prec"], 3)
    op["avg_prec_class"] = round(results["avg_prec_class"], 3)
    op["top_class"] = round(results["top_class"]["avg_prec"], 3)
    op["file_acc"] = round(results["file_acc"], 3)
    op["model"] = mod_name

    op["per_class"] = {}
    for cc in results["class_pr"]:
        op["per_class"][cc["name"]] = cc["avg_prec"]

    op_file_name = os.path.join(op_dir, mod_name + "_results.json")
    with open(op_file_name, "w") as da:
        json.dump(op, da, indent=2)


def print_results(
    model_name, mod_str, results, op_dir, class_names, file_type, title_text=""
):
    print("\nResults - " + model_name)
    print("avg_prec      ", round(results["avg_prec"], 3))
    print("avg_prec_class", round(results["avg_prec_class"], 3))
    print("top_class     ", round(results["top_class"]["avg_prec"], 3))
    print("file_acc      ", round(results["file_acc"], 3))

    print("\nSaving " + model_name + " results to: " + op_dir)
    save_summary_to_json(op_dir, mod_str, results)

    pu.plot_pr_curve(
        op_dir,
        mod_str + "_test_all_det",
        mod_str + "_test_all_det",
        results,
        file_type,
        title_text + "Detection PR",
    )
    pu.plot_pr_curve(
        op_dir,
        mod_str + "_test_all_top_class",
        mod_str + "_test_all_top_class",
        results["top_class"],
        file_type,
        title_text + "Top Class",
    )
    pu.plot_pr_curve_class(
        op_dir,
        mod_str + "_test_all_class",
        mod_str + "_test_all_class",
        results,
        file_type,
        title_text + "Per-Class PR",
    )
    pu.plot_confusion_matrix(
        op_dir,
        mod_str + "_confusion",
        results["gt_valid_file"],
        results["pred_valid_file"],
        results["file_acc"],
        class_names,
        True,
        file_type,
        title_text + "Confusion Matrix",
    )


def add_root_path_back(data_sets, ann_path, wav_path):
    for dd in data_sets:
        dd["ann_path"] = os.path.join(ann_path, dd["ann_path"])
        dd["wav_path"] = os.path.join(wav_path, dd["wav_path"])
    return data_sets


def check_classes_in_train(gt_list, class_names):
    num_gt_total = np.sum([gg["start_times"].shape[0] for gg in gt_list])
    num_with_no_class = 0
    for gt in gt_list:
        for cc in gt["class_names"]:
            if cc not in class_names:
                num_with_no_class += 1
    return num_with_no_class


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "op_dir",
        type=str,
        default="plots/results_compare/",
        help="Output directory for plots",
    )
    parser.add_argument("data_dir", type=str, help="Path to root of datasets")
    parser.add_argument(
        "ann_dir", type=str, help="Path to extracted annotations"
    )
    parser.add_argument(
        "bd_model_path", type=str, help="Path to BatDetect model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="",
        help="Path to json file used for evaluation.",
    )
    parser.add_argument(
        "--sb_ip_dir", type=str, default="", help="Path to sonobat predictions"
    )
    parser.add_argument(
        "--sb_region_classifier",
        type=str,
        default="south",
        help="Path to sonobat predictions",
    )
    parser.add_argument(
        "--td_ip_dir",
        type=str,
        default="",
        help="Path to tadarida_D predictions",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.01,
        help="IOU threshold for assigning predictions to ground truth",
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="png",
        help="Type of image to save - png or pdf",
    )
    parser.add_argument(
        "--title_text",
        type=str,
        default="",
        help="Text to add as title of plots",
    )
    parser.add_argument(
        "--rand_seed", type=int, default=2001, help="Random seed"
    )
    args = vars(parser.parse_args())

    np.random.seed(args["rand_seed"])

    if not os.path.isdir(args["op_dir"]):
        os.makedirs(args["op_dir"])

    # load the model
    params_eval = parameters.get_params(False)
    _, params_bd = du.load_model(args["bd_model_path"])

    class_names = params_bd["class_names"]
    num_classes = len(class_names) + 1  # num classes plus background class

    classes_to_ignore = ["Not Bat", "Bat", "Unknown"]
    events_of_interest = ["Echolocation"]

    # load test data
    if args["test_file"] == "":
        # load the test files of interest from the trained model
        test_sets = add_root_path_back(
            params_bd["test_sets"], args["ann_dir"], args["data_dir"]
        )
        test_sets = [
            dd for dd in test_sets if not dd["is_binary"]
        ]  # exclude bat/not datasets
    else:
        # user specified annotation file to evaluate
        test_dict = {}
        test_dict["dataset_name"] = args["test_file"].replace(".json", "")
        test_dict["is_test"] = True
        test_dict["is_binary"] = True
        test_dict["ann_path"] = os.path.join(args["ann_dir"], args["test_file"])
        test_dict["wav_path"] = args["data_dir"]
        test_sets = [test_dict]

    # load the gt for the test set
    gt_test = load_gt_data(
        test_sets, events_of_interest, class_names, classes_to_ignore
    )
    total_num_calls = np.sum([gg["start_times"].shape[0] for gg in gt_test])
    print("\nTotal number of test files:", len(gt_test))
    print(
        "Total number of test calls:",
        np.sum([gg["start_times"].shape[0] for gg in gt_test]),
    )

    # check if test contains classes not in the train set
    num_with_no_class = check_classes_in_train(gt_test, class_names)
    if total_num_calls == num_with_no_class:
        print("Classes from the test set are not in the train set.")
        assert False

    # only need the train data if evaluating Sonobat or Tadarida
    if args["sb_ip_dir"] != "" or args["td_ip_dir"] != "":
        train_sets = add_root_path_back(
            params_bd["train_sets"], args["ann_dir"], args["data_dir"]
        )
        train_sets = [
            dd for dd in train_sets if not dd["is_binary"]
        ]  # exclude bat/not datasets
        gt_train = load_gt_data(
            train_sets, events_of_interest, class_names, classes_to_ignore
        )

    #
    # evaluate Sonobat by training random forest classifier
    #
    # NOTE: Sonobat may only make predictions for a subset of the files
    #
    if args["sb_ip_dir"] != "":
        sb_meta = load_sonobat_meta(
            args["sb_ip_dir"],
            train_sets + test_sets,
            args["sb_region_classifier"],
            class_names,
        )

        preds_sb = []
        keep_inds_sb = []
        for ii, gt in enumerate(gt_test):
            sb_pred = load_sonobat_preds(gt["dataset_name"], gt["id"], sb_meta)
            if sb_pred["class_name"] != "":
                sb_pred = parse_data(
                    sb_pred, class_names, classes_to_ignore, True
                )
                sb_pred["class_probs"][
                    sb_pred["class_ids"],
                    np.arange(sb_pred["class_probs"].shape[1]),
                ] = sb_pred["det_probs"]
                preds_sb.append(sb_pred)
                keep_inds_sb.append(ii)

        results_sb = evl.evaluate_predictions(
            [gt_test[ii] for ii in keep_inds_sb],
            preds_sb,
            class_names,
            params_eval["detection_overlap"],
            params_eval["ignore_start_end"],
        )
        print_results(
            "Sonobat",
            "sb",
            results_sb,
            args["op_dir"],
            class_names,
            args["file_type"],
            args["title_text"] + " - Species - ",
        )
        print(
            "Only reporting results for",
            len(keep_inds_sb),
            "files, out of",
            len(gt_test),
        )

        # train our own random forest on sonobat features
        x_train = []
        y_train = []
        for gt in gt_train:
            pred = load_sonobat_preds(
                gt["dataset_name"], gt["id"], sb_meta, "Not Bat"
            )

            if len(pred["annotation"]) > 0:
                # compute detection overlap with ground truth to determine which are the TP detections
                assign_to_gt(gt, pred, args["iou_thresh"])
                pred = parse_data(pred, class_names, classes_to_ignore, True)
                x_train.append(pred["feats"])
                y_train.append(pred["class_ids"])

        # train random forest on tadarida predictions
        clf_sb, un_train_class = train_rf_model(
            x_train, y_train, num_classes, args["rand_seed"]
        )

        # run the model on the test set
        preds_sb_rf = []
        for gt in gt_test:
            pred = load_sonobat_preds(
                gt["dataset_name"], gt["id"], sb_meta, "Not Bat"
            )
            pred = parse_data(pred, class_names, classes_to_ignore, True)
            pred = eval_rf_model(clf_sb, pred, un_train_class, num_classes)
            preds_sb_rf.append(pred)

        results_sb_rf = evl.evaluate_predictions(
            gt_test,
            preds_sb_rf,
            class_names,
            params_eval["detection_overlap"],
            params_eval["ignore_start_end"],
        )
        print_results(
            "Sonobat RF",
            "sb_rf",
            results_sb_rf,
            args["op_dir"],
            class_names,
            args["file_type"],
            args["title_text"] + " - Species - ",
        )
        print(
            "\n\nWARNING\nThis is evaluating on the full test set, but there is only dections for a subset of files\n\n"
        )

    #
    # evaluate Tadarida-D by training random forest classifier
    #
    if args["td_ip_dir"] != "":
        x_train = []
        y_train = []
        for gt in gt_train:
            pred = load_tadarida_pred(
                args["td_ip_dir"], gt["dataset_name"], gt["id"]
            )
            # compute detection overlap with ground truth to determine which are the TP detections
            assign_to_gt(gt, pred, args["iou_thresh"])
            pred = parse_data(pred, class_names, classes_to_ignore, True)
            x_train.append(pred["feats"])
            y_train.append(pred["class_ids"])

        # train random forest on Tadarida-D predictions
        clf_td, un_train_class = train_rf_model(
            x_train, y_train, num_classes, args["rand_seed"]
        )

        # run the model on the test set
        preds_td = []
        for gt in gt_test:
            pred = load_tadarida_pred(
                args["td_ip_dir"], gt["dataset_name"], gt["id"]
            )
            pred = parse_data(pred, class_names, classes_to_ignore, True)
            pred = eval_rf_model(clf_td, pred, un_train_class, num_classes)
            preds_td.append(pred)

        results_td = evl.evaluate_predictions(
            gt_test,
            preds_td,
            class_names,
            params_eval["detection_overlap"],
            params_eval["ignore_start_end"],
        )
        print_results(
            "Tadarida",
            "td_rf",
            results_td,
            args["op_dir"],
            class_names,
            args["file_type"],
            args["title_text"] + " - Species - ",
        )

    #
    # evaluate BatDetect
    #
    if args["bd_model_path"] != "":
        # load model
        bd_args = du.get_default_bd_args()
        model, params_bd = du.load_model(args["bd_model_path"])

        # check if the class names are the same
        if params_bd["class_names"] != class_names:
            print("Warning: Class names are not the same as the trained model")
            assert False

        run_config = {
            **bd_args,
            **params_bd,
            "return_raw_preds": True,
        }

        preds_bd = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for ii, gg in enumerate(gt_test):
            pred = du.process_file(
                gg["file_path"],
                model,
                run_config,
                device,
            )
            preds_bd.append(pred)

        results_bd = evl.evaluate_predictions(
            gt_test,
            preds_bd,
            class_names,
            params_eval["detection_overlap"],
            params_eval["ignore_start_end"],
        )
        print_results(
            "BatDetect",
            "bd",
            results_bd,
            args["op_dir"],
            class_names,
            args["file_type"],
            args["title_text"] + " - Species - ",
        )

        # evaluate genus level
        class_names_genus, preds_bd_g, gt_test_g = create_genus_mapping(
            gt_test, preds_bd, class_names
        )
        results_bd_genus = evl.evaluate_predictions(
            gt_test_g,
            preds_bd_g,
            class_names_genus,
            params_eval["detection_overlap"],
            params_eval["ignore_start_end"],
        )
        print_results(
            "BatDetect Genus",
            "bd_genus",
            results_bd_genus,
            args["op_dir"],
            class_names_genus,
            args["file_type"],
            args["title_text"] + " - Genus - ",
        )
