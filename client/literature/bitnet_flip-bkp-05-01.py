
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

# def flip_probability(w_real, sigma, eps=1e-6):
#     return tf.exp(-tf.abs(w_real) / (sigma + eps))

def estimate_ckks_sigma(context,v,n_clients, dim=1000, trials=50):
    # v = np.random.randn(dim)
    v = np.array(v)
    errs = []
    for _ in range(trials):
        enc = ts.ckks_vector(context, v)
        dec = enc.decrypt()
        dec = np.array(dec)
        errs.append(dec - v)
    sigma_ckks = np.std(np.concatenate(errs))

    # σ total após soma de n clientes
    sigma_total = math.sqrt(n_clients) * sigma_ckks
    return sigma_total

# @tf.function
# def train_step(model, x, y, sigma, optimizer):
#     with tf.GradientTape() as tape:
#         logits = model(x, training=True)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
#         loss = tf.reduce_mean(loss)

#     grads = tape.gradient(loss, model.trainable_variables)

#     new_grads = []
#     for g, v in zip(grads, model.trainable_variables):
#         if "w_real" in v.name:
#             p_flip = tf.exp(-tf.abs(v) / (sigma + 1e-6))
#             alpha = 1.0 - p_flip
#             g = g * alpha
#         new_grads.append(g)

#     optimizer.apply_gradients(zip(new_grads, model.trainable_variables))
#     return loss
# Com 1 deu certo, com 2 talvez com 100 rodadas, vamos ver
def flip_confidence(w, sigma, k=2): # Alpha    
    sigma = tf.cast(sigma, w.dtype)
    k = tf.cast(k, w.dtype)
    return tf.abs(w) / (tf.abs(w) + k * sigma)

# @tf.function
# def train_step(model, x, y, sigma,server_round):
#     with tf.GradientTape() as tape:
#         preds = model(x, training=True)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
#         loss = tf.reduce_mean(loss)

#     grads = tape.gradient(loss, model.trainable_variables)
    
#     # current_k = min(0.1 + (server_round // 5) * 0.2,3)
#     # current_k = 2
#     new_grads = []
#     for w, g in zip(model.trainable_variables, grads):
        
#         if g is None:
#             new_grads.append(g)
#             continue

#         # if "kernel" in w.name:  # ajuste aqui
#         if "kernel" in w.name or "w_real" in w.name:
#             alpha = flip_confidence(tf.stop_gradient(w), sigma,k=current_k)
#             g = g * alpha
            
#         new_grads.append(g) 
#         # if "w_real" in w.name:
#         #     alpha = flip_confidence(tf.stop_gradient(w), sigma)
#         #     print(f"ALPHA MEAN: {np.mean(alpha)}")
#         #     g = g * alpha
#         #     new_grads.append(g)

#     model.optimizer.apply_gradients(zip(new_grads, model.trainable_variables))
#     return loss

# @tf.function
# def train_step(model, x, y, sigma,server_round):
#     sigma = tf.cast(sigma, tf.float32)
#     lambda_repulsion = 0.01 # Força do empurrão
#     k_width = 1.5           # Largura da barreira de potencial

#     with tf.GradientTape() as tape:
#         preds = model(x, training=True)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        
#         # --- Cálculo do Potencial de Atração ---
#         repulsion_penalty = 0.0
#         for w in model.trainable_variables:
#             if "kernel" in w.name or "w_real" in w.name:
#                 # Penaliza pesos baseada na Gaussiana centralizada no zero
#                 # Gradiente resultante empurra para longe do zero
#                 repulsion_penalty += tf.reduce_mean(tf.exp(-tf.square(w) / (2 * tf.square(k_width * sigma))))
        
#         total_loss = tf.reduce_mean(loss) + (lambda_repulsion * repulsion_penalty)

#     grads = tape.gradient(total_loss, model.trainable_variables)
    
#     # Aplica o seu amortecimento Alpha original sobre os novos gradientes
#     new_grads = []
#     current_k = min(0.1 + (server_round // 5) * 0.2,3)

#     for w, g in zip(model.trainable_variables, grads):
#         if g is not None and ("kernel" in w.name or "w_real" in w.name):
#             alpha = flip_confidence(w, sigma,k=current_k)
#             g = g * alpha
#         new_grads.append(g)

#     model.optimizer.apply_gradients(zip(new_grads, model.trainable_variables))
#     return total_loss

@tf.function
def train_step(model, x, y, sigma, server_round):
    sigma = tf.cast(sigma, tf.float32)
    server_round = tf.cast(server_round, tf.float32)
    
    # 1. K-Scheduling Sigmoidal: k fica baixo até o round 40-50
    # Isso permite o aprendizado Non-IID antes da estabilização
    k_max = 3.0
    k_midpoint = 50.0 # Round onde k começa a subir forte
    k_steepness = 0.1
    current_k = k_max / (1.0 + tf.exp(-k_steepness * (server_round - k_midpoint))) # verificar se precisa ser dinamico msm
    
    # 2. Lambda Repulsion Dinâmico
    # Só começa a empurrar os pesos após o round 20
    lambda_base = 0.01
    lambda_repulsion = tf.where(server_round > 20.0, lambda_base, 0.0)
    k_width = 1.5

    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        
        repulsion_penalty = 0.0
        if lambda_repulsion > 0:
            for w in model.trainable_variables:
                if "kernel" in w.name or "w_real" in w.name:
                    # Aplicamos o potencial apenas se houver repulsão ativa
                    repulsion_penalty += tf.reduce_mean(tf.exp(-tf.square(w) / (2 * tf.square(k_width * sigma))))
        
        total_loss = tf.reduce_mean(loss) + (lambda_repulsion * repulsion_penalty)

    grads = tape.gradient(total_loss, model.trainable_variables)
    
    new_grads = []
    for w, g in zip(model.trainable_variables, grads):
        if g is not None and ("kernel" in w.name or "w_real" in w.name):
            # O Alpha agora usa o current_k sigmoidal
            alpha = flip_confidence(w, sigma, k=current_k)
            g = g * alpha
        new_grads.append(g)

    model.optimizer.apply_gradients(zip(new_grads, model.trainable_variables))
    return total_loss

def fit_bitnetflip(self, parameters, config):
    print(f'tenseal: {self.context}')
    
    
    if config["round"]%5==0:
        v = flat_parameters(self.model.get_weights())
        self.sigma = estimate_ckks_sigma(self.context,v,self.num_clients)    
        
        
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time = time.time() - decypher_time
    
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



