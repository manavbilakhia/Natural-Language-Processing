"""Evaluation Metrics

Author: Kristina Striegnitz and Manav Bilakhia

I affirm that I have carried out my academic endeavors with full
academic honesty. [Manav Bilakhia]


Complete this file for part 1 of the project.
"""
def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    accuracy = correct / len(y_pred)
    return accuracy

def get_precision(y_pred, y_true):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    true_positives = 0
    false_positives = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            true_positives += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            false_positives += 1
    precision = true_positives / (true_positives + false_positives)
    return precision


def get_recall(y_pred, y_true):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    true_positives = 0
    false_negatives = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            true_positives += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives)
    return recall 


def get_fscore(y_pred, y_true):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = 2 * (precision * recall) / (precision + recall)
    return fscore


def evaluate(y_pred, y_true):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    accuracy = get_accuracy(y_pred, y_true) *100
    precision = get_precision(y_pred, y_true)*100
    recall = get_recall(y_pred, y_true)*100
    fscore = get_fscore(y_pred, y_true)*100
    print("Accuracy: {:.0f}".format(accuracy)+ "%")
    print("Precision: {:.0f}".format(precision)+ "%")
    print("Recall: {:.0f}".format(recall)+ "%")
    print("F-score: {:.0f}".format(fscore)+ "%")

if __name__ == "__main__":
    # Example usage
    print(get_precision([1,1,0,0,1,0,0,1,0,0], [1,1,1,1,1,0,0,0,0,0]))
    evaluate([1,1,0,0,1,0,0,1,0,0], [1,1,1,1,1,0,0,0,0,0])

