import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
)


class Metrics:
    
    def __init__(self, y_test, y_pred, y_score):
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_score = y_score 

    def compute(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred, average='macro')
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.roc = list()

        # Compute ROC curve and ROC area for each class
        n_classes = len(np.unique(self.y_test))
        y_bin = label_binarize(self.y_test, classes=np.arange(n_classes))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], self.y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), self.y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        self.roc = [fpr, tpr, roc_auc]

        return self.accuracy, self.f1, self.conf_matrix, self.roc


class Results:
    
    def __init__(self, model_name, accuracy, f1, conf_matrix, roc, grid_results):
        self.model_name = model_name
        self.accuracy = accuracy
        self.f1 = f1
        self.conf_matrix = conf_matrix
        self.roc = roc    
        self.grid_results = grid_results
    
    def plot(self) -> None:
        print(f"Resumo das métricas do modelo {self.model_name}\n")
        print(f"Accurácia média (desvio): {np.mean(self.accuracy):.3f} ({np.std(self.accuracy):.3f})")
        print(f"F1-Score média (desvio): {np.mean(self.f1):.3f} ({np.std(self.f1):.3f})")

        # Confusion Matrix
        cm_mean = np.mean(self.conf_matrix, axis=0)
        ax = sns.heatmap(cm_mean, linewidths=0.5, annot=True, fmt='g')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title('Matriz de Confusão Média')
        ax.set_xlabel('Classes preditas')
        ax.set_ylabel('Classes verdadeiras')
        plt.show()

    def save(self):
        with open(f'metrics/{self.model_name}_metrics.npy', 'wb') as f:
            np.save(f, self.accuracy, allow_pickle=True)
            np.save(f, self.f1, allow_pickle=True)
            np.save(f, self.conf_matrix, allow_pickle=True)
            np.save(f, self.roc, allow_pickle=True)
            np.save(f, self.grid_results, allow_pickle=True)

        