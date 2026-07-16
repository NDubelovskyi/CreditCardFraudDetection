from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot Method for Confusion Matrices
def plot_confusion_matrix(y_true, y_pred, title, fn):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig('../reports/figures/confusion_matrix_' + fn)
    plt.show()