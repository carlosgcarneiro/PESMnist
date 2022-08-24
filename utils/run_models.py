import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix


class RunModels:
    
    def __init__(self, models: dict) -> None:
        self.models = models
    
    def load_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y)
        
    def train(self) -> None:
        self.trained_models = dict()
        self.preds = dict()
        for name, model in self.models.items():
            self.trained_models[name] = model.fit(self.X_train, self.y_train)
            self.preds[name] = model.predict(self.X_test)
    
    def report(self, models: list[str] = None) -> None:
        if not models:
            models = self.models

        for model in models:
            print(model + '\n')
            print(classification_report(self.y_test, self.preds[model]))
        
    
    def conf_matrix(self, models: list[str] = None) -> None:
        if not models:
            models = self.models
        
        for model in models:
            print(model + '\n')
            plot_confusion_matrix(self.trained_models[model], self.X_test, self.y_test)

            
        
            
    
        