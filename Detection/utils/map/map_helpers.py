# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np

from utils.nms.nms_wrapper import apply_nms_to_test_set_results

def evaluate_detections(all_boxes, all_gt_infos, classes, use_07_metric=False, apply_mms=True, nms_threshold=0.5, conf_threshold=0.0, soft=False, confusions=None):
    '''
    Computes per-class average precision.

    Args:
        all_boxes:          shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        all_gt_infos:       a dictionary that contains all ground truth annoations in the following form:
                            {'class_A': [{'bbox': array([[ 376.,  210.,  456.,  288.,   10.]], dtype=float32), 'det': [False], 'difficult': [False]}, ... ]}
                            'class_B': [ <bbox_list> ], <more_class_to_bbox_list_entries> }
        classes:            a list of class name, e.g. ['__background__', 'avocado', 'orange', 'butter']
        use_07_metric:      whether to use VOC07's 11 point AP computation (default False)
        apply_mms:          whether to apply non maximum suppression before computing average precision values
        nms_threshold:      the threshold for discarding overlapping ROIs in nms
        conf_threshold:     a minimum value for the score of an ROI. ROIs with lower score will be discarded

    Returns:
        aps - average precision value per class in a dictionary {classname: ap}
    '''

    if apply_mms:
        print ("Number of rois before non-maximum suppression: %d" % sum([len(all_boxes[i][j]) for i in range(len(all_boxes)) for j in range(len(all_boxes[0]))]))
        nms_dets,_ = apply_nms_to_test_set_results(all_boxes, nms_threshold, conf_threshold, soft)
        print ("Number of rois  after non-maximum suppression: %d" % sum([len(nms_dets[i][j]) for i in range(len(all_boxes)) for j in range(len(all_boxes[0]))]))
    else:
        print ("Skipping non-maximum suppression")
        nms_dets = all_boxes

    aps = {}
    fp_errors = {}
    for classIndex, className in enumerate(classes):
        if className != '__background__':
            rec, prec, ap, fp_error = _evaluate_detections(classIndex, className, nms_dets, all_gt_infos, use_07_metric=use_07_metric, confusions=confusions)
            aps[className] = ap
            if fp_error is not None:
                fp_errors[className] = fp_error

    if len(fp_errors) > 0:
        return aps, fp_errors
    else:
        return aps, None

def _evaluate_detections(classIndex, className, all_boxes, all_gt_infos, overlapThreshold=0.5, use_07_metric=False, confusions=None):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''

    # parse detections for this class
    # shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
    num_images = len(all_boxes[0])
    detBboxes = []
    detImgIndices = []
    detConfidences = []
    for imgIndex in range(num_images):
        dets = all_boxes[classIndex][imgIndex]
        if dets != []:
            for k in range(dets.shape[0]):
                detImgIndices.append(imgIndex)
                # access the last element of k-th roi
                detConfidences.append(dets[k, -1])
                # the VOCdevkit expects 1-based indices
                detBboxes.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
    detBboxes = np.array(detBboxes)
    detConfidences = np.array(detConfidences)

    # compute precision / recall / ap
    rec, prec, ap, fp_error = _voc_computePrecisionRecallAp(
        className,
        all_gt_infos=all_gt_infos,
        confidence=detConfidences,
        image_ids=detImgIndices,
        BB=detBboxes,
        ovthresh=overlapThreshold,
        use_07_metric=use_07_metric,
        confusions=confusions)
    return rec, prec, ap, fp_error

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    '''
    Computes VOC AP given precision and recall.
    '''
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrecalls = np.concatenate(([0.], recalls, [1.]))
        mprecisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap

def _voc_computePrecisionRecallAp(className, all_gt_infos, confidence, image_ids, BB, ovthresh=0.5, use_07_metric=False, confusions=None):
    '''
    Computes precision, recall. and average precision

    Args:
         class_recs: ground-truth boxes info
         BB: detection roi info
    '''
    if len(BB) == 0:
        return 0.0, 0.0, 0.0, None

    # sort by confidence
    sorted_ind = np.argsort(-confidence)

    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # statics for false positive results
    # 0:localization error, 1:confusion with similiar objects
    # 2:confusion with other objects, 3:confusion with background
    # 4:duplicated detection 5:true-positive
    fp_error = None
    if confusions:
        fp_error = np.zeros(6).astype(int)
        conf = confusions[className]
        sim_classes = conf[0]
        otr_classes = conf[1]
    
    for d in range(nd):
        # ground-truth boxes for particular image
        R = all_gt_infos[className][image_ids[d]]
        # all detected rois for particular image
        bb = BB[d, :].astype(float)
        ovmax, jmax = max_overlap_with_class(className, image_ids[d], all_gt_infos, bb)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                    if fp_error is not None:
                        fp_error[5] += 1
                else:
                    # duplicate detection on the detected gt
                    fp[d] = 1.
                    if fp_error is not None:
                        fp_error[4] += 1
        else:
            fp[d] = 1.

            if fp_error is not None:
                # localization error
                if ovmax >= 0.1:
                    fp_error[0] += 1
                else:
                    # confuse with objects
                    sim_ovmax_cls, sim_ovmax = max_overlap_with_classes(list(sim_classes), image_ids[d], all_gt_infos, bb)
                    otr_ovmax_cls, otr_ovmax = max_overlap_with_classes(list(otr_classes), image_ids[d], all_gt_infos, bb)
                    
                    if sim_ovmax>=otr_ovmax and sim_ovmax>0.1:
                        fp_error[1] += 1
                    elif otr_ovmax>=sim_ovmax and otr_ovmax>0.1:
                        fp_error[2] += 1
                    else:
                        # confusion with background
                        fp_error[3] += 1
                    
    # compute precision recall
    npos = sum([len(cr['bbox']) for cr in all_gt_infos[className]])
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = computeAveragePrecision(rec, prec, use_07_metric)
    return rec, prec, ap, fp_error

def max_overlap_with_class(className, image_id, all_gt_infos, bb):
    # ground-truth boxes for particular image
    R = all_gt_infos[className][image_id]
    # access the bbox coordinate from ground-truth boxes
    BBGT = R['bbox'].astype(float)

    ovmax = -np.inf
    max_id = None
    
    if BBGT.size > 0:
        # compute overlaps concerning one single roi with all gts in the same picture
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        
        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        max_id = np.argmax(overlaps)

    return ovmax, max_id

def max_overlap_with_classes(classes, image_id, all_gt_infos, bb):
    if classes is None or len(classes)==0:
        return "", 0
    
    ovmaxes = []
    for cls in classes:
        ovmax, _ = max_overlap_with_class(cls, image_id, all_gt_infos, bb)
        ovmaxes.append(ovmax)

    max_id = np.argmax(ovmaxes)
    return classes[max_id], ovmaxes[max_id]
