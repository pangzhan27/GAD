from collections import OrderedDict
import numpy as np
import pandas as pd
import os

# 0. mAP
def thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False):
        """
        We follow (Shou et al., 2017) and adopt their perframe postprocessing method on THUMOS'14 datset.
        Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
        """

        # Simple temporal smoothing via NMS of 5-frames window
        if smooth:
            prob = np.copy(prediction)
            prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
            prob2 = np.append(prob[0, :].reshape(1, -1), prob[0: -1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob4 = np.append(prob[0: 2, :], prob[0: -2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
            prediction = np.copy(probsmooth)

        # Assign cliff diving (5) as diving (8)
        if switch:
            switch_index = np.where(prediction[:, 5] > prediction[:, 8])[0]
            prediction[switch_index, 8] = prediction[switch_index, 5]

        # Remove ambiguous (21)
        valid_index = np.where(ground_truth[:, 21] != 1)[0]

        return ground_truth[valid_index], prediction[valid_index]


def eval_perframe_map(ground_truth, prediction, class_names, ignore_index, metrics, postprocessing):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    from sklearn.metrics import average_precision_score
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result['per_class_AP'][class_name] = compute_score(
                    ground_truth[:, idx], prediction[:, idx])
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))

    return result['mean_AP']* 100




# 1. frame-wise acc, rec, prec (overall and balanced)
def eval_perframe_acc(labels, predictions, classnames, bg='background',
                      ignore='ambiguous'):  # we only consider recall with bg
    result = OrderedDict()
    predictions, labels = np.concatenate(list(predictions.values()), axis=0), np.concatenate(list(labels.values()),
                                                                                             axis=0)
    valid_index = np.where(labels != ignore)[0]
    predictions, labels = predictions[valid_index], labels[valid_index]

    # accuracy including bg
    result['accuracy'] = np.sum(predictions == labels) / len(labels) * 100

    bg_ind = labels == bg
    fg_ind = (labels != bg)
    result['bg_rec'] = np.sum(predictions[bg_ind] == labels[bg_ind]) / np.sum(bg_ind) * 100
    result['fg_rec'] = np.sum(predictions[fg_ind] == labels[fg_ind]) / np.sum(fg_ind) * 100

    bg_ind1 = predictions == bg
    fg_ind1 = (predictions != bg)
    result['bg_prec'] = np.sum((predictions[bg_ind1] == labels[bg_ind1]) & (labels[bg_ind1] == bg)) / np.sum(
        bg_ind1) * 100
    result['fg_prec'] = np.sum((predictions[fg_ind1] == labels[fg_ind1]) & (labels[fg_ind1] != bg)) / np.sum(
        fg_ind1) * 100

    result['fg_f1'] = 2 * (result['fg_prec'] * result['fg_rec']) / (result['fg_prec'] + result['fg_rec'] + 1e-16)

    def cls_rec_prec(pred, labels, selected_class):
        tp_gt = labels == selected_class
        tp_fp = pred == selected_class
        tp = tp_fp & tp_gt
        return np.sum(tp) / (np.sum(tp_gt) + 1e-8), np.sum(tp) / (np.sum(tp_fp) + 1e-8)

    bresult = OrderedDict()
    bresult['per_class_rec'] = OrderedDict()
    bresult['per_class_prec'] = OrderedDict()
    bresult['per_class_f1'] = OrderedDict()
    for idx, class_name in enumerate(classnames):
        if class_name != ignore:
            if class_name in labels:
                rec, prec = cls_rec_prec(predictions, labels, selected_class=class_name)
                f1 = 2 * (prec * rec) / (prec + rec + 1e-16)
                bresult['per_class_rec'][class_name] = rec * 100
                bresult['per_class_prec'][class_name] = prec * 100
                bresult['per_class_f1'][class_name] = f1 * 100

    bresult['mean_rec'] = np.mean(list(bresult['per_class_rec'].values()))
    bresult['mean_rec_fg'] = np.mean([v for k, v in bresult['per_class_rec'].items() if k != bg])
    bresult['mean_rec_bg'] = np.mean([v for k, v in bresult['per_class_rec'].items() if k == bg])

    bresult['mean_prec'] = np.mean(list(bresult['per_class_prec'].values()))
    bresult['mean_prec_fg'] = np.mean([v for k, v in bresult['per_class_prec'].items() if k != bg])
    bresult['mean_prec_bg'] = np.mean([v for k, v in bresult['per_class_prec'].items() if k == bg])

    bresult['mean_f1'] = np.mean(list(bresult['per_class_f1'].values()))
    bresult['mean_f1_fg'] = np.mean([v for k, v in bresult['per_class_f1'].items() if k != bg])
    bresult['mean_f1_bg'] = np.mean([v for k, v in bresult['per_class_f1'].items() if k == bg])

    return result, bresult



# 2. F1 score
def get_labels_start_end_time(frame_wise_labels, bg_class=['background']):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def f_score(recognized, ground_truth, overlap, bg_class=['background']):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    if len(y_label) == 0:
        fp += len(p_label)
        return float(tp), float(fp), 0.0

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def overlap_f1(P, Y, overlap=.1, bg_class=['background']):
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score(P[i], Y[i], overlap, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / float(TP + FP + 1e-8)
    recall = TP / float(TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    F1 = np.nan_to_num(F1)
    return precision * 100, recall * 100, F1 * 100


def f_score_ana(recognized, ground_truth, overlap, class_names, bg_class=['background']):
    num_class = len(class_names)
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = np.zeros(num_class)
    fp = np.zeros(num_class)
    fn = np.zeros(num_class)
    hits = np.zeros(len(y_label))

    if len(y_label) == 0:
        for j in range(len(p_label)):
            idx1 = class_names.index(p_label[j])
            fp[idx1] += 1
        return tp, fp, fn

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
        idx1 = class_names.index(p_label[j])

        if IoU[idx] >= overlap and not hits[idx]:
            tp[idx1] += 1
            hits[idx] = 1
        else:
            fp[idx1] += 1
    for j in range(len(y_label)):
        if hits[j] == 0:
            idx1 = class_names.index(y_label[j])
            fn[idx1] += 1

    return tp, fp, fn


def overlap_f1_macro(P, Y, class_names, overlap=.1, bg_class=["background"]) -> object:
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score_ana(P[i], Y[i], overlap, class_names, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    F1 = np.nan_to_num(F1)
    return precision * 100, recall * 100, F1 * 100


# point-wise F1
def get_sequence_from_frame_labels(frame_wise_labels, bg_class=[]):
    """Collapses a list of frame-wise labels into a list of segments
    Args:
        frame_wise_labels: corresponds to either the GT or predicted sequence of labels
            > e.g., ["pick up", "pick up", "position", "background" , "position", "position", "position", "background", "screw"]
        bg_class: list of background classes, either as label or index, for example ["background"] or [0].

    Returns:
        segments: List of segment labels,
            > e.g., ['pick up', 'position', 'screw']
        segment_starts: stores start frames for each segment
            > e.g.,  [0, 2, 8]
        segment_ends: stores end frames for each segment
            > e.g.,  [1, 6, 8]
    """

    segment_labels = []
    segment_starts = []
    segment_ends = []

    # set the first segment
    last_segment = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        segment_labels.append(frame_wise_labels[0])
        segment_starts.append(0)

    # loop through all frames to identify segments
    i = 0
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_segment:
            if frame_wise_labels[i] not in bg_class:
                segment_labels.append(frame_wise_labels[i])
                segment_starts.append(i)
            if last_segment not in bg_class:
                segment_ends.append(i)
            last_segment = frame_wise_labels[i]
    if last_segment not in bg_class:
        segment_ends.append(i)

    return segment_labels, segment_starts, segment_ends


def compute_point_level_accuracies(y_true, y_pred, threshold=0, bg_class=[], dtype='Start'):
    """Computes the point level accuracies (f1 score, precision, recall).
    Check if the action has been correctly recognized according to its distance to the action start.
    If the predicted action start is within a certain radius of the ground-truth action start, that
    prediction is considered to be a true positive.

    Reference:
    [1] WOAD: Weakly Supervised Online Action Detection in Untrimmed Videos, Gao et al., CVPR'21
    [2] StartNet: Online Detection of Action Start in Untrimmed Videos, Gao et al., ICCV'19

    Args:
        y_true: corresponds to the ground truth sequence of labels
                > e.g., ["pick up", "pick up", "position", "position"]
        y_pred: corresponds to the predicted sequence of labels
                > e.g., ["pick up", "position", "position", "screw"]
        threshold: the threshold used to determine if the action start is correctly detected
    Returns:
        f1 score: f1 score for the detection accuracy with the given threshold
        precision: precision for the detection accuracy with the given threshold
        recall: recall for the detection accuracy with the given threshold
    """
    eps = 0.0001
    assert dtype in ['Start', 'Mid'], 'unsurported dtype on point-wise F1'

    y_label, y_start, y_end = get_sequence_from_frame_labels(y_true, bg_class)
    p_label, p_start, p_end = get_sequence_from_frame_labels(y_pred, bg_class)

    # use mid
    if dtype == 'Mid':
        y_start = [(y_start[i] + y_end[i]) / 2 for i in range(len(y_start))]
        p_start = [(p_start[i] + p_end[i]) / 2 for i in range(len(p_start))]

    if len(y_label) == 0 and len(p_label) == 0:
        return 0, 0, 0

    # Count true and false positives within the overlapping area
    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        if len(y_start) == 0:
            dists = [np.inf]
        else:
            dists = np.abs(p_start[j] - np.array(y_start))
        dist_min = np.min(dists)
        jmin = np.argmin(dists)

        if dist_min <= threshold:
            if p_label[j] == y_label[jmin]:
                if not hits[jmin]:
                    tp += 1
                    hits[jmin] = 1
                else:
                    fp += 1
            else:
                fp += 1
        else:
            fp += 1

    # Compute the false negative count
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def compute_point_F1(pred_scores, gt_targets, bg_class=[], dtype='Start', fps=4, dist_ths=[1.0, 2.0]):
    out_results = []
    for dist_th in dist_ths:
        threshold = dist_th * fps
        TP, FP, FN = 0, 0, 0
        for key in gt_targets.keys():
            y_true, y_pred = gt_targets[key], pred_scores[key]
            tp, fp, fn = compute_point_level_accuracies(y_true, y_pred, threshold=threshold, bg_class=bg_class,
                                                        dtype=dtype)
            TP += tp
            FP += fp
            FN += fn

        precision = TP / float(TP + FP + 1e-8)
        recall = TP / float(TP + FN + 1e-8)
        F1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        #F1 = np.nan_to_num(F1)
        out_results.append([round(F1 * 100, 2), round(precision * 100, 2), round(recall * 100, 2)])

    avg_result = np.mean(np.array(out_results), axis=0)

    return np.concatenate((np.array(out_results), avg_result.reshape(1, -1)), axis=0)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def thumos_results(check_point, pred_scores, gt_targets, class_names, bg='background', ignore='ambiguous',
                   extra_dict={}):
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = list(set([bg, ignore]))

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls)
    mid_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 2. segment F1 score
    # p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.1, bg_class=f1_ignore_cls)
    # p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.25, bg_class=f1_ignore_cls)
    # p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.5, bg_class=f1_ignore_cls)
    #
    # b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    # b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    # b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))
    #
    # p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    # p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    # p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))
    #
    # r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    # r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    # r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore=ignore)

    for c in class_names:
        if c not in bresult['per_class_rec']:
            rec1 = -1
        else:
            rec1 = bresult['per_class_rec'][c]

        if c not in bresult['per_class_prec']:
            prec = -1
        else:
            prec = bresult['per_class_prec'][c]

        print('%-20s%-10.1f%-10.1f\n' % (c, rec1, prec))

    # result_all = {'method': check_point,
    #               'Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'], 'bg_prec': result['bg_prec'], 'fg_prec': result['fg_prec'],
    #               'mRec': bresult['mean_rec'], 'mbg_rec': bresult['mean_rec_bg'], 'mfg_rec': bresult['mean_rec_fg'],
    #               'mPrec': bresult['mean_prec'], 'mbg_prec': bresult['mean_prec_bg'], 'mfg_prec':  bresult['mean_prec_fg'],
    #               '[fr|seg]': ' | ',
    #               'F1_10': o_f1_10, 'F1_25': o_f1_25, 'F1_50': o_f1_50,
    #               'Rec_10': or_10, 'Prec_10': op_10, 'Rec_25': or_25, 'Prec_25': op_25,
    #               'Rec_50': or_50, 'Prec_50': op_50,
    #               '[ovl|bal]': ' | ',
    #               'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
    #               'mRec_10': r_f1_10, 'mPrec_10': p_f1_10,  'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
    #               'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
    #               '||Point-F1|| ': ' || ',
    #               'Start_1s(F1, prec, rec)': start_pF1[0], 's2s': start_pF1[1],
    #               '|| ': ' || ',
    #               'Mid_1s(F1, prec, rec)': mid_pF1[0], 'm2s': mid_pF1[1]
    #               }
    result_all = {'method': check_point,
                  '[Frame]- Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'],
                  '[Seg]-F1, prec, rec]': ' | ',
                  '@10': np.round(np.array([o_f1_10, op_10, or_10]), 1),
                  '@25': np.round(np.array([o_f1_25, op_25, or_25]), 1),
                  '@50': np.round(np.array([o_f1_50, op_50, or_50]), 1),
                  '||': ' || ',
                  '[Point-F1, prec, rec] Start_1s': start_pF1[0], 's2s': start_pF1[1],
                  '|| ': ' || ',
                  'Mid_1s': mid_pF1[0], 'm2s': mid_pF1[1],
                  '||| ': ' || ',
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('thumos_summary.csv'):
        head = False
    df.to_csv('thumos_summary.csv', mode='a', index=False, header=head, float_format='%.1f')


def crosstask_results(check_point, pred_scores, gt_targets, class_names, bg='background', extra_dict={}):
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = [bg]

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls)
    mid_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 2. segment F1 score

    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore='')

    result_all = {'method': check_point,
                  '[Frame]- Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'],
                  '[Seg]-F1, prec, rec]': ' | ',
                  '@10': np.round(np.array([o_f1_10, op_10, or_10]), 1),
                  '@25': np.round(np.array([o_f1_25, op_25, or_25]), 1),
                  '@50': np.round(np.array([o_f1_50, op_50, or_50]), 1),
                  '|': ' || ',
                  '[Point-F1, prec, rec] Start_1s': start_pF1[0], 's2s': start_pF1[1],
                  '|| ': ' || ',
                  'Mid_1s': mid_pF1[0], 'm2s': mid_pF1[1],
                  '||| ': ' || ',
                  }
    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('crosstask_summary.csv'):
        head = False
    df.to_csv('crosstask_summary.csv', mode='a', index=False, header=head, float_format='%.1f')


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def thumos_results_new(check_point, pred_scores, gt_targets, class_names, bg='background', ignore='ambiguous',
                       extra_dict={}):
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = list(set([bg, ignore]))

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, fps=4)
    mid_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid', fps=4)

    # 2. segment F1 score
    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.5, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore=ignore)

    # frame wise overall f1 exlude background, as it equals to accuracy
    result_all = {'method': check_point,
                  '[Frame]- Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'],
                  '-': ' * ',
                  'ov_f1': result['fg_f1'],
                  '--': ' * ',
                  'bal_F1': bresult['mean_f1'], 'bal_F1_fg': bresult['mean_f1_fg'],
                  '[Seg]': ' || ',
                  'ov-F1@10, 25, 50]': np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1),
                  '|': ' * ',
                  'bal-F1@10, 25, 50]': np.round(np.array([b_f1_10, b_f1_25, b_f1_50]), 1),
                  '[Pint]': ' || ',  # this is the overall F1
                  '[F1, prec, rec] Start_1s': start_pF1[0], 's2s': start_pF1[1],
                  '|| ': ' * ',
                  'Mid_1s': mid_pF1[0], 'm2s': mid_pF1[1],
                  '||| ': ' || ',
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('thumos_summary.csv'):
        head = False
    df.to_csv('thumos_summary.csv', mode='a', index=False, header=head, float_format='%.1f')


def crosstask_results_new(check_point, pred_scores, gt_targets, class_names, bg='background', extra_dict={}):
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = [bg]

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, fps=1)
    mid_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid', fps=1)

    # 2. segment F1 score

    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.5, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore='')

    # frame wise overall f1 exlude background, as it equals to accuracy
    result_all = {'method': check_point,
                  '[Frame]- Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'],
                  '-': ' * ',
                  'ov_f1': result['fg_f1'],
                  '--': ' * ',
                   'bal_F1': bresult['mean_f1'], 'bal_F1_fg': bresult['mean_f1_fg'],
                  '[Seg]': ' || ',
                  'ov-F1@10, 25, 50]':  np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1),
                  '|': ' * ',
                  'bal-F1@10, 25, 50]':  np.round(np.array([b_f1_10, b_f1_25, b_f1_50]), 1),
                  '[Pint]': ' || ',  # this is the overall F1
                  '[F1, prec, rec] Start_1s': start_pF1[0], 's2s': start_pF1[1],
                  '|| ': ' * ',
                 'Mid_1s': mid_pF1[0], 'm2s': mid_pF1[1],
                 '||| ': ' || ',
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('crosstask_summary.csv'):
        head = False
    df.to_csv('crosstask_summary.csv', mode='a', index=False, header=head, float_format='%.1f')


def ek100_results_new(check_point, pred_scores, gt_targets, class_names, bg='background', extra_dict={}):
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = [bg]

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, fps=4)
    mid_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid', fps=4)

    # 2. segment F1 score
    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.5, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore='')

    # result_all = {'method': check_point,
    #               '[Frame]- Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'],
    #               '[Seg]-F1, prec, rec]': ' | ',
    #               '@10': np.round(np.array([o_f1_10, op_10, or_10]), 1),
    #               '@25': np.round(np.array([o_f1_25, op_25, or_25]), 1),
    #               '@50': np.round(np.array([o_f1_50, op_50, or_50]), 1),
    #               '|': ' || ',
    #               '[Point-F1, prec, rec] Start_1s': start_pF1[0], 's2s': start_pF1[1],
    #               '|| ': ' || ',
    #               'Mid_1s': mid_pF1[0], 'm2s': mid_pF1[1],
    #               '||| ': ' || ',
    #               }

    result_all = {'method': check_point,
                  '[Frame]- Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'],
                  '-': ' * ',
                  'ov_f1': result['fg_f1'],
                  '--': ' * ',
                  'bal_F1': bresult['mean_f1'], 'bal_F1_fg': bresult['mean_f1_fg'],
                  '[Seg]': ' || ',
                  'ov-F1@10, 25, 50]': np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1),
                  '|': ' * ',
                  'bal-F1@10, 25, 50]': np.round(np.array([b_f1_10, b_f1_25, b_f1_50]), 1),
                  '[Pint]': ' || ',  # this is the overall F1
                  '[F1, prec, rec] Start_1s': start_pF1[0], 's2s': start_pF1[1],
                  '|| ': ' * ',
                  'Mid_1s': mid_pF1[0], 'm2s': mid_pF1[1],
                  '||| ': ' || ',
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('ek100_summary.csv'):
        head = False
    df.to_csv('ek100_summary.csv', mode='a', index=False, header=head, float_format='%.1f')


def ego4dgoal_results_new(check_point, pred_scores, gt_targets, class_names, bg='background', extra_dict={}):
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = [bg]

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, fps=1)
    mid_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid', fps=1)

    # 2. segment F1 score
    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, class_names, overlap=0.5, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))


    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore='')


    result_all = {'method': check_point,
                  '[Frame]- Acc': result['accuracy'], 'bg_rec': result['bg_rec'], 'fg_rec': result['fg_rec'],
                  '-': ' * ',
                  'ov_f1': result['fg_f1'],
                  '--': ' * ',
                  'bal_F1': bresult['mean_f1'], 'bal_F1_fg': bresult['mean_f1_fg'],
                  '[Seg]': ' || ',
                  'ov-F1@10, 25, 50]': np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1),
                  '|': ' * ',
                  'bal-F1@10, 25, 50]': np.round(np.array([b_f1_10, b_f1_25, b_f1_50]), 1),
                  '[Pint]': ' || ',  # this is the overall F1
                  '[F1, prec, rec] Start_1s': start_pF1[0], 's2s': start_pF1[1],
                  '|| ': ' * ',
                  'Mid_1s': mid_pF1[0], 'm2s': mid_pF1[1],
                  '||| ': ' || ',
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('ego4dgoal_summary.csv'):
        head = False
    df.to_csv('ego4dgoal_summary.csv', mode='a', index=False, header=head, float_format='%.1f')



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def thumos_results_iclr(check_point, scores, labels, class_names, bg='background', ignore='ambiguous',
                       extra_dict={}):
    # for map
    all_pred, all_gt = np.concatenate(list(scores.values()), axis=0), np.concatenate(list(labels.values()), axis=0)
    result_map = eval_perframe_map(all_gt, all_pred, class_names, 21, 'AP', thumos_postprocessing)

    # for the rest
    final_pred, final_gt = {}, {}
    for sess in scores.keys():
        pred, lbl = scores[sess], labels[sess]
        sub_pred, sub_lbl = [], []
        for i in range(len(pred)):
            sub_pred.append(class_names[np.argmax(pred[i])])
            l = np.where(lbl[i] == 1)[0]
            if 21 in l:
                sub_lbl.append(class_names[-1])
            else:
                sub_lbl.append(class_names[l[0]])
        final_pred[sess] = np.array(sub_pred)
        final_gt[sess] = np.array(sub_lbl)


    pred_scores, gt_targets = final_pred, final_gt
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = list(set([bg, ignore]))

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Start', fps=4)

    # 2. segment F1 score
    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore=ignore)

    # frame wise overall f1 exlude background, as it equals to accuracy
    result_all = {'method': check_point,
                  '[Frame]- mAP' : result_map, ' ': '*', ' Acc': result['accuracy'],  ' - ': '||',
                  '[Seg] - ov-F1@10, 25, 50]' : np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1), ' , ': '||',
                  '[Pint] - F1@Start 1, 2': np.round(np.array([start_pF1[0][0], start_pF1[1][0]]), 1),
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('thumos_summary_iclr.csv'):
        head = False
    df.to_csv('thumos_summary_iclr.csv', mode='a', index=False, header=head, float_format='%.1f')

def crosstask_results_iclr(check_point, scores, labels, class_names, bg='background', extra_dict={}):
    # for map
    all_pred, all_gt = np.concatenate(list(scores.values()), axis=0), np.concatenate(list(labels.values()), axis=0)
    result_map = eval_perframe_map(all_gt, all_pred, class_names, 0, 'AP', thumos_postprocessing)

    final_pred, final_gt = {}, {}
    for sess in scores.keys():
        pred, lbl = scores[sess], labels[sess]
        sub_pred, sub_lbl = [], []
        for i in range(len(pred)):
            sub_pred.append(class_names[np.argmax(pred[i])])
            l = np.where(lbl[i] == 1)[0]
            sub_lbl.append(class_names[l[0]])
        final_pred[sess] = np.array(sub_pred)
        final_gt[sess] = np.array(sub_lbl)

    pred_scores, gt_targets = final_pred, final_gt
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = [bg]

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, fps=1)

    # 2. segment F1 score
    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)


    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore='')

    # frame wise overall f1 exlude background, as it equals to accuracy
    result_all = {'method': check_point,
                  '[Frame]- mAP': result_map, ' Acc': result['accuracy'], ' - ': '||',
                  '[Seg] - ov-F1@10, 25, 50]': np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1), ' , ': '||',
                  '[Pint] - F1@Start 1, 2': np.round(np.array([start_pF1[0][0], start_pF1[1][0]]), 1),
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('crosstask_summary_iclr.csv'):
        head = False
    df.to_csv('crosstask_summary_iclr.csv', mode='a', index=False, header=head, float_format='%.1f')


def ek100_results_iclr(check_point, scores, labels, class_names, bg='background', extra_dict={}):
    # for map
    all_pred, all_gt = np.concatenate(list(scores.values()), axis=0), np.concatenate(list(labels.values()), axis=0)
    result_map = eval_perframe_map(all_gt, all_pred, class_names, 0, 'AP', thumos_postprocessing)

    final_pred, final_gt = {}, {}
    for sess in scores.keys():
        pred, lbl = scores[sess], labels[sess]
        sub_pred, sub_lbl = [], []
        for i in range(len(pred)):
            sub_pred.append(class_names[np.argmax(pred[i])])
            l = np.where(lbl[i] == 1)[0]
            sub_lbl.append(class_names[l[0]])
        final_pred[sess] = np.array(sub_pred)
        final_gt[sess] = np.array(sub_lbl)

    pred_scores, gt_targets = final_pred, final_gt
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = [bg]

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, fps=4)

    # 2. segment F1 score
    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore='')

    result_all = {'method': check_point,
                  '[Frame]- mAP': result_map, ' Acc': result['accuracy'], ' - ': '||',
                  '[Seg] - ov-F1@10, 25, 50]': np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1), ' , ': '||',
                  '[Pint] - F1@Start 1, 2': np.round(np.array([start_pF1[0][0], start_pF1[1][0]]), 1),
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('ek100_summary_iclr.csv'):
        head = False
    df.to_csv('ek100_summary_iclr.csv', mode='a', index=False, header=head, float_format='%.1f')

def ego4dgoal_results_iclr(check_point, scores, labels, class_names, bg='background', extra_dict={}):
    # for map
    all_pred, all_gt = np.concatenate(list(scores.values()), axis=0), np.concatenate(list(labels.values()), axis=0)
    result_map = eval_perframe_map(all_gt, all_pred, class_names, 0, 'AP', thumos_postprocessing)

    final_pred, final_gt = {}, {}
    for sess in scores.keys():
        pred, lbl = scores[sess], labels[sess]
        sub_pred, sub_lbl = [], []
        for i in range(len(pred)):
            sub_pred.append(class_names[np.argmax(pred[i])])
            l = np.where(lbl[i] == 1)[0]
            sub_lbl.append(class_names[l[0]])
        final_pred[sess] = np.array(sub_pred)
        final_gt[sess] = np.array(sub_lbl)

    pred_scores, gt_targets = final_pred, final_gt
    sess_pred, sess_gt = list(pred_scores.values()), list(gt_targets.values())
    f1_ignore_cls = [bg]

    # 1. point-F1
    start_pF1 = compute_point_F1(pred_scores, gt_targets, bg_class=f1_ignore_cls, fps=1)

    # 2. segment F1 score
    op_10, or_10, o_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    op_25, or_25, o_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    op_50, or_50, o_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result, bresult = eval_perframe_acc(gt_targets, pred_scores, class_names, bg=bg, ignore='')

    result_all = {'method': check_point,
                  '[Frame]- mAP': result_map, ' Acc': result['accuracy'], ' - ': '||',
                  '[Seg] - ov-F1@10, 25, 50]': np.round(np.array([o_f1_10, o_f1_25, o_f1_50]), 1), ' , ': '||',
                  '[Pint] - F1@Start 1, 2': np.round(np.array([start_pF1[0][0], start_pF1[1][0]]), 1),
                  }

    for k in extra_dict.keys():
        result_all[k] = extra_dict[k]

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('ego4dgoal_summary_iclr.csv'):
        head = False
    df.to_csv('ego4dgoal_summary_iclr.csv', mode='a', index=False, header=head, float_format='%.1f')

