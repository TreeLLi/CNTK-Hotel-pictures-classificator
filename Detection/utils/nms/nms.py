# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def nms(dets, ovr_thresh, soft=False, conf_thresh=0.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        if soft:
            inds = np.where(ovr > ovr_thresh)[0]
            reduc = order[inds+1]
            ovrs = ovr[inds]
            scores[reduc] = scores[reduc] * (1-ovrs)
            order = scores.argsort()[::-1]
            del order[np.where(order==i)[0]]
            if scores[order[0]] < conf_thresh:
                break
            
        else:
            inds = np.where(ovr <= ovr_thresh)[0]
            order = order[inds + 1]

    return keep
