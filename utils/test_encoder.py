from sklearn.preprocessing import StandardScaler

from keras import Model, Sequential
from keras.layers import Dense



class Encoder():

    def __init__(self, layer_configs=None, loss='binary_crossentropy',
                optimizer='adagrad', metrics=['accuracy'], epochs = 5, 
                batch_size=128):

        if layer_configs is None:
            layer_configs =  [{'n_neurons': 784, 'activation': 'linear'},
                              {'n_neurons': 196, 'activation': 'sigmoid'}]
        
        self.epochs = epochs
        self.batch_size=batch_size

        self.enc = Sequential()

        self.enc.add(Dense(layer_configs[0]['n_neurons'],
                        input_shape=(layer_configs[0]['n_neurons'],),
                        activation=layer_configs[0]['activation']))

        # input layer until encoder layer
        for layer_config in layer_configs[1:-1]:
            self.enc.add(Dense(layer_config['n_neurons'], 
                                activation=layer_config['activation']))

        # encoder layer
        self.enc.add(Dense(layer_configs[-1]['n_neurons'],
                        activation=layer_configs[-1]['activation'],
                        name='encoded'))

        # encoder layer until last layer
        layer_configs.reverse()
        for layer_config in layer_configs[1:]:
            self.enc.add(Dense(layer_config['n_neurons'],
                                activation=layer_config['activation']))

        self.enc.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # plot_model(self.enc, path+'encoder_compress.png', show_shapes=True)
        
    
    def set_params(self, layer_configs, epochs=5, batch_size=128):
        self.layer_configs = layer_configs
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self, X, y=None):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.enc.fit(
            X,X,
            batch_size=self.batch_size,
            epochs = self.epochs,
        )

        return self

    def transform(self, X, y=None):
        hidden_layer = self.enc.get_layer('encoded').output
        encoder = Model(inputs=self.enc.input, outputs=hidden_layer)

        return encoder.predict(X)
        