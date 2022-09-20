from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adamax


def MLP(input_shape, n_classes, n_neurons,
        activation='relu', loss='categorical_crossentropy',
        optimizer=Adam, lr=0.001, metrics=['accuracy']):
    
    model = Sequential()
    
    first_layer_neurons = n_neurons[0]
    model.add(Dense(first_layer_neurons, 
                input_shape=input_shape,
                activation=activation))
    
    for n_neuron in n_neurons[1:]:
        model.add(Dense(n_neuron, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(loss=loss, optimizer=optimizer(lr), metrics=metrics)

    return model