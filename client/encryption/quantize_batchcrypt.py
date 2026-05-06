import numpy as np
from .aciq import ACIQ
from .complement import true2two, two2true

# ATENÇÃO À ESCOLHA DO ALPHA AQUI (Leia a explicação abaixo)
# ALPHA = 5.0
ALPHA = 600000.0
precision = 5

# ==========================================
# FUNÇÕES CORE DO PAILLIER (Complemento de 2)
# ==========================================

def quantize_padding_batchcrypt(value, alpha, quan_bits, num_clients):
    # clipping
    value = np.clip(value, -alpha, alpha)

    # quantizing
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (pow(2, quan_bits - 1) - 1.0) / (alpha * num_clients)
    value = unsigned_value * sign

    # stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int)
    
    # Adiciona bits de padding e converte para complemento de 2
    padding_bits = int(np.ceil(np.log2(num_clients)))
    res_val = []
    for elem in value:
        res_val.append(true2two(elem, padding_bits, quan_bits))
    return res_val

def unquantize_padding_batchcrypt(value, alpha, quan_bits, num_clients):
    value = np.array(value, dtype=object)
    padding_bits = int(np.ceil(np.log2(num_clients)))

    # Extrai o complemento de 2 de volta para o valor real
    res_value = []
    for elem in value:
        if elem == 0:
            res_value.append(elem)
            continue
        elem = two2true(elem, padding_bits, quan_bits)
        res_value.append(elem)
    
    value = np.array(res_value, dtype=object)

    # unquantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
    value = unsigned_value * sign 
    return value

# ==========================================
# WRAPPERS DE QUANTIZAÇÃO
# ==========================================

def quantize_batchcrypt(plains, element_bits, num_clients):
    quantized = quantize_padding_batchcrypt(plains, ALPHA, element_bits, num_clients)
    return np.array(quantized)

def unquantize_batchcrypt(plains, element_bits, num_clients):
    print(f"[DEBUG] unquantize_batchcrypt: element_bits={element_bits}, num_clients={num_clients}")
    unquantized = unquantize_padding_batchcrypt(plains, ALPHA, element_bits, num_clients)
    # Proteção contra o bug do ufunc rint do NumPy:
    unquantized = np.array(unquantized, dtype=float)
    return np.round(unquantized, precision)

# ==========================================
# FUNÇÕES DE BIT-PACKING (Exclusivas do Batchcrypt)
# ==========================================

def batch_padding_batchcrypt(array, max_bits, elem_bits, batch_size):
    elem_bits = elem_bits + 4
    array = array.tolist()
    
    if len(array) % batch_size != 0:
        pad_zero_nums = batch_size - len(array) % batch_size
        array += [0] * pad_zero_nums
  
    batch_nums = len(array) // batch_size
    batch_list = []
    mod = pow(2, elem_bits)

    for b in range(batch_nums):
        tmp = 0
        for i in range(batch_size):
            tmp *= mod
            tmp += array[i + b * batch_size]
        batch_list.append(tmp)

    return np.array(batch_list).astype(object)

def unbatching_padding_batchcrypt(array, elem_bits, batch_size):
    print(f"[DEBUG] unbatching_padding_batchcrypt: elem_bits={elem_bits}, batch_size={batch_size}")
    elem_bits = elem_bits + 4
    res = []
    mask = pow(2, elem_bits) - 1

    for item in array:
        item = int(item)
        if item == 0:
            tmp = [0] * batch_size
            res.extend(tmp)
            continue
        tmp = []
        for i in range(batch_size):
            num = item & mask
            item >>= elem_bits
            tmp.append(num)
        tmp.reverse()
        res += tmp
        
    return np.array(res)