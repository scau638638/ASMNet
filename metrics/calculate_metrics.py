import sys
sys.path.append('../')
from metrics.binary_confusion_matrix import get_binary_confusion_matrix, get_threshold_binary_confusion_matrix
from metrics.binary_statistical_metrics import get_accuracy, get_true_positive_rate, get_true_negative_rate, \
    get_precision, get_f1_socre, get_iou


# np.seterr(divide='ignore', invalid='ignore')


def calculate_metrics( preds, targets, device, fixed_threshold=True):
    # 固定阈值
    if fixed_threshold:
        curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
            input_=preds, target=targets, device=device, pixel=2,
            threshold=0.5,
            reduction='sum')

        # curr_acc = get_accuracy(true_positive=curr_TP,
        #                         false_positive=curr_FP,
        #                         true_negative=curr_TN,
        #                         false_negative=curr_FN)

        curr_recall = get_true_positive_rate(true_positive=curr_TP,
                                             false_negative=curr_FN)

        # curr_specificity = get_true_negative_rate(false_positive=curr_FP,
        #                                           true_negative=curr_TN)

        curr_precision = get_precision(true_positive=curr_TP,
                                       false_positive=curr_FP)

        curr_f1_score = get_f1_socre(true_positive=curr_TP,
                                     false_positive=curr_FP,
                                     false_negative=curr_FN)

        curr_iou = get_iou(true_positive=curr_TP,
                           false_positive=curr_FP,
                           false_negative=curr_FN)

        # curr_auroc = get_auroc(preds, targets)
        threshold = 0.5
    else:
        # 非固定阈值
        fusion_mat = get_threshold_binary_confusion_matrix(input_=preds, target=targets, device=device, pixel=2,
                                                           reduction='sum')

        max_metrics = [0, 0, 0, 0]
        threshold = 1/100

        for i, (curr_TP, curr_FP, curr_TN, curr_FN) in enumerate(fusion_mat):
            recall = get_true_positive_rate(true_positive=curr_TP, false_negative=curr_FN)

            precision = get_precision(true_positive=curr_TP, false_positive=curr_FP)

            f1_score = get_f1_socre(true_positive=curr_TP, false_positive=curr_FP, false_negative=curr_FN)

            iou = get_iou(true_positive=curr_TP, false_positive=curr_FP, false_negative=curr_FN)

            if f1_score > max_metrics[2]:
                max_metrics[0] = recall
                max_metrics[1] = precision
                max_metrics[2] = f1_score
                max_metrics[3] = iou
                threshold = (i + 1) / 100

        curr_recall = max_metrics[0]
        curr_precision = max_metrics[1]
        curr_f1_score = max_metrics[2]
        curr_iou = max_metrics[3]

    # return (curr_acc, curr_recall, curr_specificity, curr_precision,
    #         curr_f1_score, curr_iou, curr_auroc)
    return (curr_recall, curr_precision,
            curr_f1_score, curr_iou, threshold)

