import pandas as pd

from sklearn.metrics import classification_report


class RunModels:
    
    def __init__(self, models: dict) -> None:
        self.models = models
        self.scores = dict()
    
    def load_data(self, X_train: pd.DataFrame, X_test: pd.Series,
                  y_train: pd.DataFrame, y_test: pd.Series) -> None:
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def train(self) -> None:
        self.trained_models = dict()
        self.preds = dict()
        for name, model in self.models.items():
            self.trained_models[name] = model.fit(self.X_train, self.y_train)
            self.preds[name] = model.predict(self.X_test)
    
    def accuracy(self, models: list[str] = None) -> dict:
        if not models:
            models = self.models.keys()

        for model in models:
            self.scores[model] = self.trained_models[model].score(
                self.X_test,
                self.y_test,
            )
        
        return self.scores

    def report(self, models: list[str] = None) -> None:
        if not models:
            models = self.models.keys()

        for model in models:
            print(model + '\n')
            print(classification_report(self.y_test, self.preds[model]))
        