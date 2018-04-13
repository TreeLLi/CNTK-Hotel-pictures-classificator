
import os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_path, "../.."))

from FasterRCNN.config import cfg
from utils.annotations.annotations_helper import parse_class_map_file

dataset_path = os.path.join(curr_path, "../"+cfg["CNTK"].MAP_FILE_PATH)
eva_path = os.path.join(curr_path, "../../FasterRCNN/Output/evaluations.txt")
eva_file = open(eva_path, 'w+')
eva_file.close()

fp_errors_infos = ["Localization", "Similiar", "Others", "Background", "Duplicated"]

def log_fp_errors(className, fp_errors):
    with open(eva_path, 'a') as eva_file:
        eva_file.write(className + ":\n")
        ratios = fp_erros / np.sum(fp_errors)
        for idx, amount in enumerate(fp_errors):
            info = fp_errors_infos[idx]
            ratio = ratios[idx]
            line = "{:>15}: {}({:.2f})".format(info, amount, ratio)
            eva_file.write(line)

def confusion_classes(class_name):
    confusions = _load_confusions_file()

    sim_cls = confusions[class_name]
    classes = _load_classes_file()
    otr_cls = [cls for idx, cls in enumerate(classes) if ((cls not in sim_cls) and cls!=class_name)]

    return sim_cls, otr_cls

def _load_classes_file():
    classes_file_path = os.path.join(dataset_path, cfg["CNTK"].CLASS_MAP_FILE)
    classes = parse_class_map_file(classes_map_file_path)
    # remove the background class
    del classes[0]
    return classes

def _load_confusions_file():
    classes = _load_classes_file()
    confusions = {cls:set() for cls in classes}
    conf_file_path = os.path.join(dataset_path, cfg["CNTK"].CONFUSION_FILE)
    with open(conf_file_path, 'r') as conf_file:
        for line in conf_file:
            sim_cls = line.split(":")
            for cls in sim_cls:
                if cls in classes:
                    for s_cls in sim_cls:
                        if s_cls!=cls:
                            confusion[cls].add(s_cls)

    return list(confusions)
