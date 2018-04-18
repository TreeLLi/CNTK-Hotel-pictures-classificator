
import os, sys
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_path, "../.."))

#from FasterRCNN.config import cfg
from utils.annotations.annotations_helper import parse_class_map_file

# dataset_path = os.path.join(curr_path, "../"+cfg["CNTK"].MAP_FILE_PATH)
# #dataset_path = os.path.join(curr_path, "../../../DataSets/HotailorPOC2")
# eva_path = os.path.join(curr_path, "../../FasterRCNN/Output/evaluations.txt")
# eva_file = open(eva_path, 'w+')
# eva_file.close()

fp_errors_infos = ["Localization", "Similiar", "Others", "Background", "Duplicated"]

def log_fp_errors(fp_errors, output_file):
    for className, fp_error in fp_errors:
        with open(output_file, 'w+') as output:
            output.write(className + ":\n")
            total = np.sum(fp_error)
            total_tp = fp_error[-1]
            fp_error = fp_error[:-1]
            total_fp = np.sum(fp_error)
            tp_ratio = total_tp / float(total)
            fp_ratio = total_fp / float(total)
            output.write("total: {:d}, tp: {:d}({:.2f}), fp: {:d}({:.2f})\n".format(total, total_tp,tp_ratio, total_fp, fp_ratio))
            ratios = fp_error / total_fp
            for idx, amount in enumerate(fp_error):
                info = fp_error_infos[idx]
                ratio = ratios[idx]
                line = "{:>15}: {:d}({:.2f})\n".format(info, amount, ratio)
                output.write(line)

# def confusion_classes(class_name):
#     confusions = _load_confusions_file()

#     sim_cls = confusions[class_name]
#     classes = _load_classes_file()
#     otr_cls = [cls for idx, cls in enumerate(classes) if ((cls not in sim_cls) and cls!=class_name)]

#     return sim_cls, otr_cls

def confusions_map(classes, conf_file):
    conf = _load_confusions_file(classes, conf_file)

    for cls, cf in conf.items():
        sim_cls = cf[0]
        otr_cls = [cls for idx, cls in enumerate(classes) if ((cls not in sim_cls) and cls!=class_name)]
        cf.append(otr_cls)
    return conf

def _load_confusions_file(classes, conf_file_path):
    confusions = {cls:[set()] for cls in classes}
    with open(conf_file_path, 'r') as conf_file:
        lines = conf_file.read().splitlines()
        for line in lines:
            sim_cls = line.split(":")
            for cls in sim_cls:
                if cls in classes:
                    for s_cls in sim_cls:
                        if s_cls!=cls:
                            confusions[cls][0].add(s_cls)

    return confusions
