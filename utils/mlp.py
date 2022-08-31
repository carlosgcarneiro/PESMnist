from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def MLP(input_shape, n_classes, n_neurons,
        activation='relu', loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy']):
                
    
    model = Sequential()
    
    first_layer_neurons = n_neurons.pop(0)
    model.add(Dense(first_layer_neurons, 
                input_shape=input_shape,
                activation=activation))
    
    for n_neuron in n_neurons:
        model.add(Dense(n_neuron, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model
            