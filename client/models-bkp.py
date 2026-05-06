import tensorflow as tf
import numpy as np
from keras import layers, models, Sequential
from typing import Tuple
import tensorflow as tf

# class BitDense(tf.keras.layers.Layer):
#     def __init__(self, units):
#         super().__init__()
#         self.units = units

#     def build(self, input_shape):
#         self.w_real = self.add_weight(
#             shape=(input_shape[-1], self.units),
#             initializer="glorot_uniform",
#             trainable=True
#         )

#     def call(self, x):
#         # Binarização sem estado zero
#         w_bin = tf.where(self.w_real >= 0, 1.0, -1.0)

#         # Escala média (BitNet-style)
#         alpha = tf.reduce_mean(tf.abs(self.w_real), axis=0, keepdims=True)
#         w_bin = alpha * w_bin

#         # STE
#         w_bin = tf.stop_gradient(w_bin - self.w_real) + self.w_real

#         return tf.matmul(x, w_bin)


# def create_bitnet(input_shape, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=input_shape),
#         tf.keras.layers.Flatten(),
#         BitDense(64),
#         tf.keras.layers.ReLU(),
#         BitDense(32),
#         tf.keras.layers.ReLU(),
#         BitDense(16),
#         tf.keras.layers.ReLU(),
#         tf.keras.layers.Dense(num_classes, activation="softmax")
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(1e-3),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#     return model
@tf.custom_gradient
def sign_with_ste(x):
    # Forward pass: Binariza para -1 ou 1
    out = tf.sign(x)
    # Garante que não existam zeros (converte 0 para -1)
    out = tf.where(tf.equal(out, 0), -tf.ones_like(out), out)

    def grad(dy, variables=None):
        # STE: O gradiente passa apenas se o peso latente estiver em [-1, 1]
        # Isso evita que os pesos latentes cresçam infinitamente
        grad_x = dy * tf.cast(tf.abs(x) <= 1.0, tf.float32)
        if variables is not None:
            return grad_x, [grad_x] # Retorna gradiente para x e para o kernel
        return grad_x

    return out, grad

class BinaryDense(layers.Layer): # Alterado para Layer para evitar conflitos com Dense
    def __init__(self, units, use_bias=True, **kwargs):
        super(BinaryDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform", # Melhor para evitar flips iniciais
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer="zeros",
                trainable=True
            )
        super().build(input_shape)

    def call(self, inputs):
        # Binarização do kernel no Forward pass
        binary_kernel = sign_with_ste(self.kernel)
        # binary_kernel = tf.stop_gradient(binary_kernel - self.kernel) + self.kernel
        
        # Multiplicação binarizada
        output = tf.matmul(inputs, binary_kernel)
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return output

class BinaryActivation(layers.Layer):
    def call(self, inputs):
        return sign_with_ste(inputs)
    
    
    
# # --- Modelo pequeno de BitNet ---
def create_bitnet(input_shape, num_classes, alpha=1.0):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        BinaryDense(256),
        layers.BatchNormalization(),
        BinaryActivation(),
        
        BinaryDense(256),
        layers.BatchNormalization(),
        BinaryActivation(),
        
        BinaryDense(128),
        layers.BatchNormalization(),
        BinaryActivation(),   
     
        BinaryDense(64),
        layers.BatchNormalization(),
        BinaryActivation(), 
       layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model





def create_dnn(input_shape:Tuple[int,int], num_classes:int,seed:int):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape[1:])),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(64,  activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(32,  activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed)),
        tf.keras.layers.Dense(num_classes, activation='softmax'),

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def CNN_fmnist_tf(input_shape:Tuple[int,int], num_classes:int):
    model = models.Sequential()
    
    # Layer 1
    model.add(layers.Conv2D(16, kernel_size=(5, 5), padding='same', input_shape=(input_shape[1:])))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 2
    model.add(layers.Conv2D(32, kernel_size=(3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Layer 3
    model.add(layers.Conv2D(64, kernel_size=(3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def residual_block(inputs, filters, stride=1):
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding="same", use_bias=False)(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def ResNet20(input_shape, num_classes):
    inputs = tf.keras.Input(shape=(input_shape[1:]))

    # Initial Conv Layer
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # First stack of residual blocks (16 filters)
    for _ in range(3):
        x = residual_block(x, 16)

    # Second stack of residual blocks (32 filters)
    x = residual_block(x, 32, stride=2)
    for _ in range(2):
        x = residual_block(x, 32)

    # Third stack of residual blocks (64 filters)
    x = residual_block(x, 64, stride=2)
    for _ in range(2):
        x = residual_block(x, 64)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def create_lenet5(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(*input_shape[1:],1), padding="same"))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='tanh'))
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
   # Convolutional Block 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Convolutional Block 2
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Convolutional Block 3
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    
    # Fully Connected Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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