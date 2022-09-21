from sklearn.preprocessing import StandardScaler

from keras import Model
from keras.layers import Dense, Input


class Encoder():
    
    def __init__(self, reduce_amount=0.5):
        self.reduce_amount = reduce_amount
    
    def fit(self, X, y=None):
        self.input_size = X.shape[1]
        self.inputs = Input(shape=(self.input_size,))

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        hidden_size = int(round((X.shape[1]*self.reduce_amount),0))
        
        encoded = Dense(hidden_size, activation='sigmoid', name='encoded')(self.inputs)
        outputs = Dense(self.input_size, activation='linear')(encoded)
        
        self.enc = Model(self.inputs, outputs)
        
        self.enc.compile(optimizer='adagrad', loss='binary_crossentropy')
        self.enc.fit(
            X,X,
            batch_size=64,
            epochs=20,
        )

        return self

    def transform(self, X, y=None):
        hidden_layer = self.enc.get_layer('encoded').output
        encoder = Model(inputs=self.enc.input, outputs=hidden_layer)

        return encoder.predict(X)