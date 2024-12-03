from nltk import edit_distance as ed
from MakeGS import load_gs


def norm_ld(s1, s2):
    """Calculate the percentage difference between two strings"""

    lev_dist = ed(s1, s2)  # Get the edit distance

    l1 = len(s1)
    l2 = len(s2)
    max_dif = max(l1, l2)  # Find the length of the larger of the two strings (this is the max possible edit distance)

    lev_norm = (lev_dist/max_dif)*100  # Normalise the edit distance, then render as a percentage of difference
    # lev_norm = round(lev_norm, 2)  # Round to two decimal places

    return lev_norm


def ed_compare(str1, str2):
    """Compares two strings, predicts whether they're related based on edit distance"""

    lev_norm = norm_ld(str1, str2)
    if lev_norm >= 1.0:
        result = "Related"
    else:
        result = "Unrelated"

    return result


def compare_glosses(glosses, method="ED"):
    """
    Compares two sets of glosses, predicts whether each gloss is related or unrelated
    """

    labels = [g[2] for g in glosses]

    # If using the edit distance method
    if method == "ED":
        results = [ed_compare(g[0], g[1]) for g in glosses]
    # If using any other method (none ready yet)
    else:
        results = labels

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for check_no, check_result in enumerate(results):
        if check_result == labels[check_no] and check_result == "Related":
            tp += 1
        elif check_result == labels[check_no] and check_result == "Unrelated":
            tn += 1
        elif check_result != labels[check_no] and check_result == "Related":
            fp += 1
        elif check_result != labels[check_no] and check_result == "Unrelated":
            fn += 1

    total = tp + tn + fp + fn
    accuracy = round(((tp + tn)/total), 2)
    precision = round((tp/(tp + fp)), 2)
    recall = round((tp/(tp + fn)), 2)
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = round(2 * ((precision * recall) / (precision + recall)), 2)

    return (
        f"        TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        f"        Accuracy: {accuracy},\n        Precision: {precision},\n"
        f"        Recall: {recall},\n        f-measure: {f_measure}"
    )


if __name__ == "__main__":

    dev_set = load_gs("Gold Standard Dev.pkl")
    print(compare_glosses(dev_set))

    """
    The following results were recorded using the edit distance method to compare glosses from the development set.
    Any normalised edit distance higher than the cutoff (n) is considered to indicate related glosses, so the results
    are given for various values for n.
    
    For this task, the f-measure is more important than the other metrics.
    
    n = 1
        TP: 490, FP: 1265, TN: 1, FN: 282
        Accuracy: 0.24,
        Precision: 0.28,
        Recall: 0.63,
        f-measure: 0.39
    n = 3
        TP: 489, FP: 1265, TN: 1, FN: 283
        Accuracy: 0.24,
        Precision: 0.28,
        Recall: 0.63,
        f-measure: 0.39
    n = 5
        TP: 484, FP: 1265, TN: 1, FN: 288
        Accuracy: 0.24,
        Precision: 0.28,
        Recall: 0.63,
        f-measure: 0.39
    n = 10
        TP: 473, FP: 1265, TN: 1, FN: 299
        Accuracy: 0.23,
        Precision: 0.27,
        Recall: 0.61,
        f-measure: 0.37
    n = 15
        TP: 456, FP: 1259, TN: 7, FN: 316
        Accuracy: 0.23,
        Precision: 0.27,
        Recall: 0.59,
        f-measure: 0.37
    n = 20
        TP: 444, FP: 1258, TN: 8, FN: 328
        Accuracy: 0.22,
        Precision: 0.26,
        Recall: 0.58,
        f-measure: 0.36
    n = 30
        TP: 402, FP: 1247, TN: 19, FN: 370
        Accuracy: 0.21,
        Precision: 0.24,
        Recall: 0.52,
        f-measure: 0.33
    n = 40
        TP: 356, FP: 1218, TN: 48, FN: 416
        Accuracy: 0.2,
        Precision: 0.23,
        Recall: 0.46,
        f-measure: 0.31
    n = 50
        TP: 253, FP: 1180, TN: 86, FN: 519
        Accuracy: 0.17,
        Precision: 0.18,
        Recall: 0.33,
        f-measure: 0.23
    n = 60
        TP: 133, FP: 1085, TN: 181, FN: 639
        Accuracy: 0.15,
        Precision: 0.11,
        Recall: 0.17,
        f-measure: 0.13
    n = 70
        TP: 64, FP: 885, TN: 381, FN: 708
        Accuracy: 0.22,
        Precision: 0.07,
        Recall: 0.08,
        f-measure: 0.07
    n = 80
        TP: 15, FP: 569, TN: 697, FN: 757
        Accuracy: 0.35,
        Precision: 0.03,
        Recall: 0.02,
        f-measure: 0.02
    n = 90
        TP: 2, FP: 189, TN: 1077, FN: 770
        Accuracy: 0.53,
        Precision: 0.01,
        Recall: 0.0,
        f-measure: 0.0
    n = 100
        TP: 0, FP: 37, TN: 1229, FN: 772
        Accuracy: 0.6,
        Precision: 0.0,
        Recall: 0.0,
        f-measure: 0
    """
