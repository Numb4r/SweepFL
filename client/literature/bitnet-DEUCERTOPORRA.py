
import time 

import sys 
import tenseal as ts 
import numpy as np
from .set_parameters import he_parameters_to_model
from  client_utils import flat_parameters
from  client_logs import write_train_logs
import tensorflow as tf
import math
import os
import pickle as pk




def repulsion_penalty(model, delta):
    penalties = []
    for w in model.trainable_variables:
        if ("kernel" in w.name) or ("w_real" in w.name):
            # hinge: (max(0, delta - |w|))^2
            penalties.append(tf.reduce_mean(tf.square(tf.nn.relu(delta - tf.abs(w)))))
    return tf.add_n(penalties) if penalties else tf.constant(0.0, tf.float32)

def update_confidence(g, sigma, lr, k):
    # confiança baseada no TAMANHO DO PASSO (lr*g) vs ruído sigma
    sigma = tf.cast(sigma, g.dtype)
    lr    = tf.cast(lr, g.dtype)
    k     = tf.cast(k, g.dtype)
    step  = tf.abs(lr * g)
    return step / (step + k * sigma + 1e-12)

@tf.function
def train_step(model, x, y, sigma_avg, server_round):
    sigma_avg = tf.cast(sigma_avg, tf.float32)
    server_round = tf.cast(server_round, tf.float32)

    # hiperparams/schedules
    lambda_repulsion = tf.where(server_round > 20.0, 1e-4, 0.0)  # fraco
    delta = 2.0 * sigma_avg                                      # zona "ruidosa"
    k = tf.constant(0.4, tf.float32)                             # gating do update

    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        )

        rep = repulsion_penalty(model, tf.cast(delta, tf.float32))
        total_loss = loss + tf.cast(lambda_repulsion, tf.float32) * tf.cast(rep, tf.float32)

    grads = tape.gradient(total_loss, model.trainable_variables)

    # clip básico (evita explosão)
    grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]

    # learning rate atual (tensor)
    lr = model.optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        lr = lr(model.optimizer.iterations)
    lr = tf.cast(lr, tf.float32)

    new_grads = []
    for w, g in zip(model.trainable_variables, grads):
        if g is None:
            new_grads.append(None)
            continue

        # gating só em pesos "reais" de interesse
        if ("kernel" in w.name) or ("w_real" in w.name):
            # alpha é por-elemento
            alpha = update_confidence(g, sigma_avg, lr, k)
            alpha = tf.clip_by_value(alpha, 0.05, 1.0)  # não zera tudo
            g = g * tf.cast(alpha, g.dtype)

        new_grads.append(g)

    model.optimizer.apply_gradients(zip(new_grads, model.trainable_variables))
    return total_loss


def ema_blend_weights(w_prev, w_new, gamma):
    # gamma pequeno = mais suavização (menos drift)
    return [(1-gamma)*a + gamma*b for a, b in zip(w_prev, w_new)]
def fit_bitnetflip(self, parameters, config):
    print(f'tenseal: {self.context}')
    self.sigma = 0.003505564108215556

    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time = time.time() - decypher_time
    
    # no fit, após set_weights do global:
    gamma = 0.3  # 0.1~0.5; comece em 0.3 nessa config
    w_new = self.model.get_weights()
    
    if getattr(self, "prev_global", None) is None:
        self.prev_global = [np.array(x, copy=True) for x in w_new]
    else:
        w_smooth = ema_blend_weights(self.prev_global, w_new, gamma)
        self.model.set_weights(w_smooth)
        self.prev_global = [np.array(x, copy=True) for x in w_smooth]
        
        
    
    batch_size = 32 # Ou o valor que você preferir
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (self.x_train, self.y_train)
    ).shuffle(1000).batch(batch_size)
    
    
    train_time = time.time()
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for x_batch, y_batch in train_dataset:
        # 1. Executa o passo de treino
        loss_value = train_step(self.model, x_batch,y_batch, self.sigma,config['round'])
        # 2. Registra a perda atual
        epoch_loss_avg.update_state(loss_value)
        
        # 3. Calcula a acurácia para este batch
        predictions = self.model(x_batch, training=False)
        epoch_accuracy.update_state(y_batch, predictions)


    train_time = time.time() - train_time

    # acc        = np.mean(history.history['accuracy'])
    # loss       = np.mean(history.history['loss'])
    loss = epoch_loss_avg.result().numpy()
    acc = epoch_accuracy.result().numpy()
    loss = float(loss)
    acc= float(acc)
    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time        = time.time()
    

    if self.homomorphic:
        flatted_parameters = flat_parameters(trained_parameters)
       
        # print(f'flatted_parameters max: {max(flatted_parameters)}')
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
            
        he_parameters      = ts.ckks_vector(self.context, flatted_parameters)
        
        he_parameters      = he_parameters.serialize()
        model_size         = sys.getsizeof(he_parameters)
        topk_mask          =  '' 
    
    
    
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time,sigma=self.sigma)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
    # if self.cid == 0:
    #     folder = f'logs/models/model_client0_{self.solution}_{self.technique}_{self.poly_modulus_degree}_{self.coeff_mod_bit_sizes}_{self.global_scale}_{self.key_type}/'
    #     os.makedirs(f'{folder}', exist_ok=True)  

    #     with open(f"{folder}/round_{config['round']}.pkl", 'wb') as f:
    #         pk.dump(trained_parameters, f)
    
    return fit_msg



