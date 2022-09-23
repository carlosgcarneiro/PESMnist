from sklearn.preprocessing import StandardScaler

from keras import Model
from keras.layers import Dense, Input


class Encoder():
    
    def __init__(self, input_size, output_size):
        self.output_size = output_size
        self.input_size = input_size
        self.inputs = Input(shape=(self.input_size,))

        hidden_size = int(round((self.output_size),0))
        
        encoded = Dense(hidden_size, activation='linear', name='encoded')(self.inputs)
        outputs = Dense(self.input_size, activation='sigmoid')(encoded)
        
        self.enc = Model(self.inputs, outputs)
        
        self.enc.compile(optimizer='adagrad', loss='binary_crossentropy')
    
    def fit(self, X, y=None):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.enc.fit(
            X,X,
            batch_size=128,
            epochs=5,
        )

        return self

    def transform(self, X, y=None):
        hidden_layer = self.enc.get_layer('encoded').output
        encoder = Model(inputs=self.enc.input, outputs=hidden_layer)

        return encoder.predict(X)