import tensorflow as tf
import numpy as np
from keras import layers, models, Sequential
from typing import Tuple
import tensorflow as tf

# RESNET
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------
# Bloco Residual Básico (CIFAR version)
# ---------------------------------------
def residual_block(x, filters, stride=1):
    shortcut = x
    
    # Primeira conv
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Segunda conv
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Ajuste do atalho se mudar dimensão
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# ---------------------------------------
# ResNet-20 CIFAR
# ---------------------------------------
def create_resnet20(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Camada inicial
    x = layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1 (16 filtros)
    for _ in range(3):
        x = residual_block(x, 16, stride=1)

    # Stage 2 (32 filtros)
    x = residual_block(x, 32, stride=2)
    for _ in range(2):
        x = residual_block(x, 32, stride=1)

    # Stage 3 (64 filtros)
    x = residual_block(x, 64, stride=2)
    for _ in range(2):
        x = residual_block(x, 64, stride=1)

    # Classificador
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def create_simple_cnn(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # <<< AQUI
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_dnn(input_shape:Tuple[int,int], num_classes:int,seed:int) :
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape)),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(128, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        # tf.keras.layers.Dense(64,  activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        # tf.keras.layers.Dense(32,  activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(256, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(256,  activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(128,  activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(64,  activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(num_classes, activation='softmax'),

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model



def create_lenet5(input_shape, num_classes):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        6,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='tanh',
        padding="same",
        input_shape=input_shape
    ))

    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(
        16,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='tanh',
        padding='valid'
    ))

    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='tanh'))
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def flat_parameters(parameters):
    flat_params = []

    for param in parameters:
        flat_params.extend(param.flatten())

    return flat_params


def reshape_parameters(self, decrypted_parameters):
    reshaped_parameters = []

    for layer in self.model.get_weights():
        reshaped_parameters.append(np.reshape(decrypted_parameters[:layer.size], layer.shape))
        decrypted_parameters = decrypted_parameters[layer.size:]

    return reshaped_parameters
def reshape_model(flatted_packs,a,sentinel=0):
    model =[]
    # print(a)
    
    if type(a) != int and type(a)!=float:
        # print(f'len:{len(a)}')
        for item in a:
            # print(f'item {item},  {type(item)},sentinel {sentinel}')
            
            m,sentinel=reshape_model(flatted_packs,item,sentinel)
            # print(sentinel)
            model.append(m)
    else:
    
        sentinel+=1 
        # print(f'a: {a}')
        return flatted_packs[sentinel-1],sentinel
    return model,sentinel