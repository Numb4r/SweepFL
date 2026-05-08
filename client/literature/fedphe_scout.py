import time 
import numpy as np
import pickle 
import math

import tensorflow as tf

from .set_parameters import he_packs_to_model
from  client_utils import  cypher_packs, flat_parameters, packing, get_size
from  client_logs import write_train_logs
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


def fit_fedphe_scout(self, parameters, config):
    sigma_table={
        # "8192 [60, 17, 60] 17": 0.003505564108215556,
        # "8192 [40, 17, 40] 17": 0.003491085298324097,
        # "8192 [30, 18, 30] 18": 0.0017336158168782118,
        "8192 [60, 17, 60] 17": 0.0034658750805435096,
        "8192 [40, 17, 40] 17": 0.0034830180017280687,
        "8192 [30, 18, 30] 18": 0.0017471863170911575,
        "8192 [60, 40, 40, 60] 40": 4.151022044597767e-10,
        
        
    }
    poly = self.key_config["poly_modulus_degree"]
    coeffs = str(self.key_config["coeff_mod_bit_sizes"])
    log_scale = int(math.log2(self.key_config["global_scale"]))
    key_table = f"{poly} {coeffs} {log_scale}"
    print(sigma_table.keys())
    print(f"Key table: {key_table}")
    self.sigma = sigma_table.get(key_table)
    print(f"Using sigma: {self.sigma} for key config: {key_table}")
    
    
    
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        if self.packing:
            he_packs_to_model(self, config)    
    decypher_time = time.time() - decypher_time
    
    
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
    batch_size = 4
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        .shuffle(1000)
        .batch(batch_size)
    )
    train_time = time.time()
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
    train_time = time.time() - train_time

    loss = float(epoch_loss_avg.result().numpy())
    acc  = float(epoch_accuracy.result().numpy())

    print(
        f"[round {r}] loss={loss:.4f} acc={acc:.4f} "
        f"alpha={float(epoch_alpha.result().numpy()):.4f} "
        f"clip%={float(epoch_clip.result().numpy()):.4f} "
        f"|step|/sigma={float(epoch_ratio.result().numpy()):.4f}"
    )
    trained_parameters = self.model.get_weights()
    he_parameters      = []
    model_size         = 0
    chunk_size         = 0
    qnt_packing        = 0

    flatten_time_clock    = 0.0
    flatten_time_cpu      = 0.0
    pack_time_clock       = 0.0
    pack_time_cpu         = 0.0
    mask_select_time_clock = 0.0
    mask_select_time_cpu  = 0.0
    encrypt_time_clock    = 0.0
    encrypt_time_cpu      = 0.0
    serialize_time_clock  = 0.0
    serialize_time_cpu    = 0.0

    if self.packing:
        chunk_size = self.key_config["poly_modulus_degree"] // 2

        flatten_time_clock = time.time()
        flatten_time_cpu   = time.process_time()
        flatted_parameters = flat_parameters(trained_parameters)
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
        flatten_time_cpu   = time.process_time() - flatten_time_cpu
        flatten_time_clock = time.time() - flatten_time_clock

        pack_time_clock = time.time()
        pack_time_cpu   = time.process_time()
        packed_parameters = packing(flatted_parameters, chunk_size=chunk_size)
        pack_time_cpu   = time.process_time() - pack_time_cpu
        pack_time_clock = time.time() - pack_time_clock

        mask_select_time_clock = time.time()
        mask_select_time_cpu   = time.process_time()
        mask = get_topk_mask(packed_parameters, self.percentage)
        mask_select_time_cpu   = time.process_time() - mask_select_time_cpu
        mask_select_time_clock = time.time() - mask_select_time_clock

        encrypt_time_clock = time.time()
        encrypt_time_cpu   = time.process_time()
        cyphered_packs = cypher_packs(self, packed_parameters, mask)
        encrypt_time_cpu   = time.process_time() - encrypt_time_cpu
        encrypt_time_clock = time.time() - encrypt_time_clock

        serialize_time_clock = time.time()
        serialize_time_cpu   = time.process_time()
        he_parameters = pickle.dumps(cyphered_packs)
        serialize_time_cpu   = time.process_time() - serialize_time_cpu
        serialize_time_clock = time.time() - serialize_time_clock

        model_size  = get_size(cyphered_packs)
        qnt_packing = len(packed_parameters) * self.percentage

    times = {
        'decrypt_time_clock':      decypher_time,
        'decrypt_time_cpu':        0.0,
        'train_time_clock':        train_time,
        'train_time_cpu':          0.0,
        'flatten_time_clock':      flatten_time_clock,
        'flatten_time_cpu':        flatten_time_cpu,
        'pack_time_clock':         pack_time_clock,
        'pack_time_cpu':           pack_time_cpu,
        'mask_select_time_clock':  mask_select_time_clock,
        'mask_select_time_cpu':    mask_select_time_cpu,
        'encrypt_time_clock':      encrypt_time_clock,
        'encrypt_time_cpu':        encrypt_time_cpu,
        'serialize_time_clock':    serialize_time_clock,
        'serialize_time_cpu':      serialize_time_cpu,
    }
    write_train_logs(
        self,
        config['round'],
        loss,
        acc,
        model_size,
        times,
        qnt_packing=qnt_packing,
        size_pack=chunk_size,
        sigma=self.sigma,
    )

    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, mask)
    
    return fit_msg


def get_topk_mask(packs, percentage):
    n = len(packs)
    if n == 0:
        return []

    # Garante faixa válida
    percentage = float(percentage)
    percentage = max(0.0, min(1.0, percentage))

    topk = int(math.ceil(n * percentage))
    if topk <= 0:
        return [0] * n

    # Calcula "score" por pack
    avg = np.array([np.mean(np.abs(pack)) for pack in packs], dtype=np.float64)

    # Remove NaN/Inf da disputa (NaN vira -inf, +inf fica +inf)
    avg = np.nan_to_num(avg, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    # Pega índices dos topk maiores
    idx = np.argsort(avg)[::-1][:topk]  # topk maiores
    idx.sort()  # opcional: manter ordem crescente de índices

    mask = np.zeros(n, dtype=np.int8)
    mask[idx] = 1
    return mask.tolist()
def get_slide_window_mask(round,packs,size_window=0.1,stride=3,percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
        stride = math.ceil(size_window*stride)
    
    array_z = np.zeros(len(packs))
    start = round*stride
    end = round*stride+size_window
    for i in range(start, end):
        array_z[i % len(array_z)] = 1
    return array_z

def get_robin_round_mask(round,packs,size_window,percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
    return get_slide_window_mask(round,packs,size_window,size_window)
def get_pondering_random_mask(packs,k,weights_packs,penalty=0.5,reward=0.1,percentage=False):

    n = len(packs)
    if percentage:
        k = math.ceil(k*n)
    probabilities = weights_packs / weights_packs.sum()
    mask = np.zeros(n)

    
    chosen = np.random.choice(range(n), size=k, replace=False, p=probabilities)
    
    for i in range(n):
        if i in chosen:
            weights_packs[i] *= penalty  
            mask[i] = 1
        else:
            weights_packs[i] += reward  # Aumenta o peso dos não escolhidos

    
    weights_packs = np.maximum(weights_packs, 0.01)
    return mask

