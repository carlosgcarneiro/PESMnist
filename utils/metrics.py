import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

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

        self.plot_cm()
        self.plot_roc()
    
    def plot_cm(self):
        fig, ax = plt.subplots()
        # Confusion Matrix
        cm_mean = np.mean(self.conf_matrix, axis=0)
        ax = sns.heatmap(cm_mean, linewidths=0.5, annot=True, fmt='g', ax=ax, cmap = sns.cm.rocket_r)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(f'Matriz de Confusão Média\n{self.model_name}')
        ax.set_xlabel('Classes preditas')
        ax.set_ylabel('Classes verdadeiras')
        
        plt.savefig(f'graphs/{self.model_name}_cm.png')
        plt.show()

        
    def plot_roc(self):
        fig, axs = plt.subplots(2,5, figsize=(20,10))
        #roc_curve
        tprs = dict()
        aucs = dict()
        mean_fpr = np.linspace(0, 1, 100)

        for cls in range(10):
            tprs_fold = []
            aucs_fold = []
            for fold in range(10):
                interp_tpr = np.interp(mean_fpr, self.roc[:][fold][0].get(cls), self.roc[:][fold][1].get(cls))
                interp_tpr[0] = 0.0
                tprs_fold.append(interp_tpr)
                aucs_fold.append( self.roc[:][fold][2].get(cls))
            tprs[cls] = tprs_fold
            aucs[cls] = aucs_fold

        colors = cm.rainbow(np.linspace(0, 1, 11))
        for cls in range(10):
            for fold, color in zip(range(10), colors):
                axs[cls//5, cls%5].plot(self.roc[:][fold][0].get(cls), self.roc[:][fold][1].get(cls), color=color, alpha=0.6,lw=1,
                        label='ROC - Fold {0} (AUC = {1:0.2f})'
                        ''.format(fold, self.roc[:][fold][2].get(cls)))

            axs[cls//5, cls%5].plot([0, 1], [0, 1], 'k--', lw=1)

            mean_tpr = np.mean(tprs[cls], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs[cls])

            axs[cls//5, cls%5].plot(
                mean_fpr,
                mean_tpr,
                color=colors[-1],
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )
            
            std_tpr = np.std(tprs[cls], axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            axs[cls//5, cls%5].fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.6,
                label=r"$\pm$ 1 std. dev.",
            )
        
            axs[cls//5, cls%5].axis(xmin=-0.05, xmax=1.0, ymin=0.0, ymax=1.05)
            axs[cls//5, cls%5].set_xlabel('Taxa de Falso Positivo', fontsize='small')
            axs[cls//5, cls%5].set_ylabel('Taxa de Verdadeiro Positivo', fontsize='small')
            axs[cls//5, cls%5].set_title(f'Classe "{cls}"')
            axs[cls//5, cls%5].legend(loc="lower right", prop={'size': 6})

        fig.suptitle(f'{self.model_name} - ROC', fontsize='x-large')

        plt.savefig(f'graphs/{self.model_name}_roc.png')
        plt.show()

    def save(self):
        with open(f'metrics/{self.model_name}_metrics.npy', 'wb') as f:
            np.save(f, self.accuracy, allow_pickle=True)
            np.save(f, self.f1, allow_pickle=True)
            np.save(f, self.conf_matrix, allow_pickle=True)
            np.save(f, self.roc, allow_pickle=True)
            np.save(f, self.grid_results, allow_pickle=True)