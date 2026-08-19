from sklearn.metrics import (
    accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
)
def calulate_accuracy(y_true,y_pred):
    """
    Calculate Accuracy
    """
    return accuracy_score(y_true,y_pred)

def calculate_precision(y_true,y_pred):
    """
    Calculate Precision
    """
    return precision_score(y_true,y_pred,zero_division=0)

def calculate_recall(y_true,y_pred):
    """
    Calculate Recall-score
    """
    return recall_score(y_true,y_pred,zero_division=0)

def calculate_f1(y_true,y_pred):
    """
    Calculate F1-Score
    """
    return f1_score(y_true,y_pred,zero_division=0)

def calculate_all_metrics(y_true,y_pred):
    """
    Calculate all metrics
    """
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,zero_division=0)
    recall=recall_score(y_true,y_pred,zero_division=0)
    f1=f1_score(y_true,y_pred,zero_division=0)

    return{
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f1
    }

def print_metrics(y_true, y_pred):
    """
    Print model performance.
    """

    metrics = calculate_all_metrics(
        y_true,
        y_pred
    )

    print("--------------------------------")
    print("MODEL PERFORMANCE")
    print("--------------------------------")

    print(
        f"Accuracy  : {metrics['accuracy']:.2f}"
    )

    print(
        f"Precision : {metrics['precision']:.2f}"
    )

    print(
        f"Recall    : {metrics['recall']:.2f}"
    )

    print(
        f"F1 Score  : {metrics['f1_score']:.2f}"
    )


def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


def get_classification_report(y_true, y_pred):
    """
    Generate classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        zero_division=0
    )


def check_model_performance(
    accuracy,
    threshold=0.70
):
    """
    Check whether model performance
    has degraded.
    """

    if accuracy < threshold:

        return {
            "status": "ALERT",
            "message": "Model performance degraded. Retraining required."
        }

    return {
        "status": "HEALTHY",
        "message": "Model is performing well."
    }

