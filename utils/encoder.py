from sklearn.preprocessing import StandardScaler

from keras import Model
from keras.layers import Dense, Input


class Encoder():
    
    def __init__(self, input_size, reduce_amount=0.5):
        self.reduce_amount = reduce_amount
        self.input_size = input_size
        self.inputs = Input(shape=(self.input_size,))

        hidden_size = int(round((self.input_size*self.reduce_amount),0))
        
        encoded = Dense(hidden_size, activation='sigmoid', name='encoded')(self.inputs)
        outputs = Dense(self.input_size, activation='linear')(encoded)
        
        self.enc = Model(self.inputs, outputs)
        
        self.enc.compile(optimizer='adagrad', loss='binary_crossentropy')
    
    def fit(self, X, y=None):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

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