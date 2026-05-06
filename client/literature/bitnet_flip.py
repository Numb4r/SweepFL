import time
import sys
import tenseal as ts
import numpy as np
from .set_parameters import he_parameters_to_model
from client_utils import flat_parameters
from client_logs import write_train_logs
import tensorflow as tf
import math
import os
import pickle as pk


# =========================
#  Técnica: penalties + trust-region step + gating vs sigma
# =========================

def repulsion_penalty(model, delta):
    penalties = []
    for w in model.trainable_variables:
        if ("kernel" in w.name) or ("w_real" in w.name):
            # hinge: (max(0, delta - |w|))^2
            penalties.append(tf.reduce_mean(tf.square(tf.nn.relu(delta - tf.abs(w)))))
    return tf.add_n(penalties) if penalties else tf.constant(0.0, tf.float32)


def confidence_from_step(step, sigma, k):
    """Confiança por elemento usando magnitude do step vs sigma"""
    sigma = tf.cast(sigma, step.dtype)
    k     = tf.cast(k, step.dtype)
    a = tf.abs(step)
    return a / (a + k * sigma + 1e-12)


@tf.function
def train_step(model, x, y, sigma_avg, server_round):
    # métricas internas (por batch)
    alpha_sum = tf.constant(0.0, tf.float32)
    alpha_cnt = tf.constant(0.0, tf.float32)
    clip_sum  = tf.constant(0.0, tf.float32)
    step_ratio_sum = tf.constant(0.0, tf.float32)

    sigma_avg = tf.cast(sigma_avg, tf.float32)
    server_round = tf.cast(server_round, tf.float32)

    # =========================
    # Schedules (somente técnica)
    # =========================
    # começa repulsão a partir do round 5
    lambda_repulsion = tf.where(server_round >= 5.0, 2e-4, 0.0)

    # zona morta hinge (quanto maior, mais empurra pesos pra fora)
    delta = 3.0 * sigma_avg

    # gating k por round (mais conservador no início)
    k = tf.where(server_round <= 5.0, 1.0,
        tf.where(server_round <= 20.0, 0.7, 0.4))
    k = tf.cast(k, tf.float32)

    # =========================
    # Forward + loss + penalty
    # =========================
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        )

        rep = repulsion_penalty(model, tf.cast(delta, tf.float32))
        total_loss = loss + tf.cast(lambda_repulsion, tf.float32) * tf.cast(rep, tf.float32)

    grads = tape.gradient(total_loss, model.trainable_variables)

    # =========================
    # LR tensor (suporta schedule)
    # =========================
    lr = model.optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        lr = lr(model.optimizer.iterations)
    lr = tf.cast(lr, tf.float32)

    # =========================
    # Trust-region por step (cap proporcional ao sigma)
    # =========================
    # Ajuste principal para estabilidade em baixa escala:
    #  - c_step menor = mais conservador
    #  - c_step maior = mais agressivo
    c_step = tf.constant(3.0, tf.float32)   # típico 2.0 ~ 5.0
    step_cap = c_step * sigma_avg

    # =========================
    # Aplica técnica em grads (mantém pareamento correto com None)
    # =========================
    new_grads = []

    for w, g in zip(model.trainable_variables, grads):
        if g is None:
            new_grads.append(None)
            continue

        if ("kernel" in w.name) or ("w_real" in w.name):
            g32 = tf.cast(g, tf.float32)
            step = lr * g32

            # 1) clip do step ANTES do alpha (trust-region)
            step = tf.clip_by_value(step, -step_cap, step_cap)

            # % clipped ANTES de reduzir por alpha (senão fica falso baixo)
            clipped_mask = tf.cast(tf.greater_equal(tf.abs(step), step_cap), tf.float32)
            clip_sum += tf.reduce_mean(clipped_mask)

            # 2) gating suave baseado no step vs sigma
            alpha = confidence_from_step(step, sigma_avg, k)
            alpha = tf.clip_by_value(alpha, 0.05, 1.0)
            step = step * alpha

            # métricas
            alpha_sum += tf.reduce_mean(alpha)
            alpha_cnt += 1.0
            step_ratio_sum += tf.reduce_mean(tf.abs(step) / (sigma_avg + 1e-12))

            # volta pra gradiente
            g = tf.cast(step / (lr + 1e-12), g.dtype)

        new_grads.append(g)

    # =========================
    # Global norm clip preservando Nones (não quebra zip)
    # =========================
    idxs = []
    g_list = []
    for i, g in enumerate(new_grads):
        if g is not None:
            idxs.append(i)
            g_list.append(g)

    if tf.size(idxs) > 0:
        clipped_list, _ = tf.clip_by_global_norm(g_list, 1.0)
        final_grads = list(new_grads)
        for j, i in enumerate(idxs):
            final_grads[i] = clipped_list[j]
    else:
        final_grads = new_grads

    model.optimizer.apply_gradients(zip(final_grads, model.trainable_variables))

    mean_alpha = alpha_sum / (alpha_cnt + 1e-12)
    mean_clip  = clip_sum  / (alpha_cnt + 1e-12)
    mean_ratio = step_ratio_sum / (alpha_cnt + 1e-12)

    return total_loss, mean_alpha, mean_clip, mean_ratio


def ema_blend_weights(w_prev, w_new, gamma):
    # gamma pequeno = mais suavização (menos drift)
    return [(1 - gamma) * a + gamma * b for a, b in zip(w_prev, w_new)]


def fit_bitnetflip(self, parameters, config):
    print(f"tenseal: {self.context}")

    # sigma FIXO DA CONFIG (ckks 8192 [60,17,60] scale 2**17)
    # self.sigma = 0.003505564108215556
    # sigma FIXO DA CONFIG  ckks 8192 30,18,30 scale 2**12
    sigma_table={
        # "8192 [60, 17, 60] 17": 0.003505564108215556,
        # "8192 [40, 17, 40] 17": 0.003491085298324097,
        # "8192 [30, 18, 30] 18": 0.0017336158168782118,
        "8192 [60, 17, 60] 17": 0.0034658750805435096,
        "8192 [40, 17, 40] 17": 0.0034830180017280687,
        "8192 [30, 18, 30] 18": 0.0017471863170911575,
        "8192 [60, 40, 40, 60] 40": 4.151022044597767e-10,
        "8192 [30, 17, 30] 17":0.003498611988352308,
        "8192 [50, 18, 50] 18":0.0017337328636884846
        
        
    }
    poly = self.key_config["poly_modulus_degree"]
    coeffs = str(self.key_config["coeff_mod_bit_sizes"])
    log_scale = int(math.log2(self.key_config["global_scale"]))
    key_table = f"{poly} {coeffs} {log_scale}"
    print(sigma_table.keys())
    print(f"Key table: {key_table}")
    try:
        self.sigma = sigma_table.get(key_table)
    except:
        print(f"Erro retriving sigma {key_table}")
        exit()
    print(f"Using sigma: {self.sigma} for key config: {key_table}")

    # =========================
    # Decifra global -> aplica no modelo
    # =========================
    decypher_time_clock = time.time()
    decypher_time_process = time.process_time()
    if len(config["he"]) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time_process = time.process_time() - decypher_time_process
    decypher_time_clock = time.time() - decypher_time_clock
    
    print("PASSOU PELA DECIFRA")
    # =========================
    # Suavização global (EMA dos pesos globais recebidos)
    # =========================
    r = config["round"]
    gamma = 0.1 if r < 5 else (0.2 if r < 20 else 0.25)

    w_new = self.model.get_weights()
    if getattr(self, "prev_global", None) is None:
        self.prev_global = [np.array(x, copy=True) for x in w_new]
    else:
        w_smooth = ema_blend_weights(self.prev_global, w_new, gamma)
        self.model.set_weights(w_smooth)
        self.prev_global = [np.array(x, copy=True) for x in w_smooth]
    print("PREPARANDO PARA TREINO LOCAL")
    # =========================
    # Treino local
    # =========================
    batch_size = 4
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        .shuffle(1000)
        .batch(batch_size)
    )
    print("COMEÇANDO TREINO LOCAL")
    train_time_clock = time.time()
    train_time_process = time.process_time()
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # métricas da técnica (média por época)
    epoch_alpha = tf.keras.metrics.Mean()
    epoch_clip  = tf.keras.metrics.Mean()
    epoch_ratio = tf.keras.metrics.Mean()

    for x_batch, y_batch in train_dataset:
        loss_value, mean_alpha, mean_clip, mean_ratio = train_step(
            self.model, x_batch, y_batch, self.sigma, config["round"]
        )

        epoch_loss_avg.update_state(loss_value)
        epoch_alpha.update_state(mean_alpha)
        epoch_clip.update_state(mean_clip)
        epoch_ratio.update_state(mean_ratio)

        predictions = self.model(x_batch, training=False)
        epoch_accuracy.update_state(y_batch, predictions)

    for v in self.model.trainable_variables:
        _ = v.numpy()
    train_time_process = time.process_time() - train_time_process
    train_time_clock = time.time() - train_time_clock
    print("TERMINADO TREINO LOCAL")
    loss = float(epoch_loss_avg.result().numpy())
    acc  = float(epoch_accuracy.result().numpy())

    # (debug útil) — se quiser desligar depois, remove
    print(
        f"[round {r}] loss={loss:.4f} acc={acc:.4f} "
        f"alpha={float(epoch_alpha.result().numpy()):.4f} "
        f"clip%={float(epoch_clip.result().numpy()):.4f} "
        f"|step|/sigma={float(epoch_ratio.result().numpy()):.4f}"
    )

    # =========================
    # Cifra parâmetros treinados
    # =========================
    trained_parameters = self.model.get_weights()
    he_parameters = []
    cypher_time = time.time()
    model_size = 0
    print("COMECANDO A CIFRAGEM")
    cypher_time_clock        = time.time()
    cypher_time_process       = time.process_time()
    if self.homomorphic:
        flatted_parameters = flat_parameters(trained_parameters)

        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)

        he_parameters = ts.ckks_vector(self.context, flatted_parameters)
        he_parameters = he_parameters.serialize()

        model_size = sys.getsizeof(he_parameters)

    cypher_time_process       = time.process_time() - cypher_time_process
    cypher_time_clock = time.time() - cypher_time_clock
    print("TERMINANDO CIFRAGEM")
    # =========================
    # Logs
    # =========================
    # write_train_logs(
    #     self,
    #     config["round"],
    #     loss,
    #     acc,
    #     model_size,
    #     train_time,
    #     cypher_time,
    #     decypher_time,
    #     sigma=self.sigma,
    # )
    # def write_train_logs(self, rnd, loss, acc, model_size, times, qnt_packing=0,size_pack=0,sigma=None):
    topk_mask = ""
    
    times={
        'train_time_clock': train_time_clock,
        'train_time_process': train_time_process,
        
        'cypher_time_clock': cypher_time_clock,
        'cypher_time_process': cypher_time_process,
        
        'decypher_time_clock': decypher_time_clock,
        'decypher_time_process': decypher_time_process
    }
    write_train_logs(self, config['round'], loss, acc, model_size, times,sigma=self.sigma)
    
    fit_msg = self.create_fit_msg(train_time_process, acc, loss, model_size, he_parameters, topk_mask)
    print(fit_msg)
    return fit_msg
