def plot_prc(y_test, prob, model_name = ""):
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    
    precision, recall, tresh = precision_recall_curve(y_test, prob)
    auc = average_precision_score(y_test, prob)
    
    plt.plot(precision, recall, label = "{} AUPRC = {}".format(model_name, auc.round(2)))
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall Curve")
    plt.legend()

def plot_roc(y_test, prob, model_name = ""):
    
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    
    fpr, recall, tresh = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    
    plt.plot(fpr, recall, label = "{} AUC = {}".format(model_name, auc.round(2)))
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel("FPR")
    plt.ylabel("Recall")
    plt.title("ROC Curve")
    plt.legend()
