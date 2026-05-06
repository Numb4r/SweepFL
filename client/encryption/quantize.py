# import numpy as np
# from .aciq import ACIQ
# from .complement import true2two, two2true



# precision = 5

# # get aciq alpha
# def get_alpha_r_max(plains,elements_bits, num_clients):
#     list_min = np.min(plains)
#     list_max = np.max(plains)
#     list_size = len(plains)

#     aciq  = ACIQ(elements_bits)
#     alpha = aciq.get_alpha_gaus(list_min,list_max,list_size)
#     r_max = alpha * num_clients
#     return alpha, r_max

# # quantize and padding  return two_complement
# def quantize_padding(value, alpha, quan_bits, num_clients):
#     # clipping
#     value = np.clip(value, -alpha, alpha)

#     # quantizing
#     sign = np.sign(value)
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * (pow(2, quan_bits - 1) - 1.0) / (alpha * num_clients)
#     value = unsigned_value * sign

#     # stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)
    
#     # add padding bits and show with complement
#     padding_bits = int(np.ceil(np.log2(num_clients)))
#     res_val = []
#     for elem in value:
#         res_val.append(true2two(elem, padding_bits, quan_bits))
#     return res_val

# # unquantize and return true value
# def unquantize_padding(value, alpha, quan_bits, num_clients):
#     value = np.array(value)
#     # stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)
#     padding_bits = int(np.ceil(np.log2(num_clients)))
#     #alpha *= pow(2,padding_bits)

#     # extract 2's complement to true value
#     res_value = []
#     for elem in value:
#         # if elem < 0:
#         #     raise ValueError("Overflow {}".format(elem))
#         #     # print("Overflow {}".format(elem))
#         #     # res_value.append(0)
#         #     # continue
#         if elem == 0:
#             res_value.append(elem)
#             continue
#         #elem &= ((1 << (quan_bits-1)) - 1)
#         elem = two2true(elem, padding_bits, quan_bits)
#         res_value.append(elem)
#     value = res_value

#     # unquantize
#     sign = np.sign(value)
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
#     value = unsigned_value * sign 
#     return value

# def quan_no_compl(value, quan_bits, num_clients):
#     alpha = 1.0
#     # clipping
#     value = np.clip(value, -alpha, alpha)

#     # quantizing
#     sign = np.sign(value)
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * (pow(2, quan_bits - 1) - 1.0) / (alpha * num_clients)
#     value = unsigned_value * sign

#     # stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)
    
#     return value

# def unquan_no_compl(value, quan_bits, num_clients):
#     alpha = 1.0
#     value = np.array(value)
#     # stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)
 
#     # unquantize
#     sign = np.sign(value)
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
#     value = unsigned_value * sign 
#     return value  

# def quantize_postive(value,quan_bits,num_clients):
#     alpha = 1.0
#     # clipping
#     value = np.clip(value, -alpha, alpha)

#     value = np.add(value,1.0)
#     print(np.max(value))

#     # quantizing
#     value = value * (pow(2, quan_bits - 1) - 1.0) / (alpha * num_clients)

#     # stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)

#     return value

# def unquantize_postive(value,quan_bits,num_clients,idx_weights):
#     value = np.array(value)
#     # stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)
#     alpha = 1.0

#     # unquantize
#     value = value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
#     #idx_weights = [0.14,0.35,0.45,0.5] 

#     offset = np.sum(idx_weights)
#     value = np.add(value, -1 * offset)
#     return value

# # quantize function
# def quantize(plains,element_bits,num_clients):
#     #alpha,r_max = get_alpha_r_max(plains,element_bits,num_clients)
#     alpha = 5.0
#     quantized = quantize_padding(plains,alpha,element_bits,num_clients)
#     return np.array(quantized)

# # unquantize function
# def unquantize(plains, element_bits, num_clients, alpha = 5.0):
#     unquantized = unquantize_padding(plains, alpha, element_bits, num_clients)
#     unquantized = np.array(unquantized)
#     return np.round(unquantized, precision)

# # batch elems function
# def batch_padding(array, max_bits, elem_bits,batch_size):
#     elem_bits = elem_bits + 4
#     array = array.tolist()
#     #batch_size = max_bits // elem_bits
#     if len(array) % batch_size != 0:
#         pad_zero_nums = batch_size - len(array) % batch_size
#         array += [0] * pad_zero_nums
  
#     # how many batches is needed
#     batch_nums = len(array) // batch_size
  
#     # carry batches
#     batch_list = []
#     mod = pow(2, elem_bits)

#     # for each batch
#     for b in range(batch_nums):
#         tmp = 0
#         # for each elem
#         for i in range(batch_size):
#             tmp *= mod
#             tmp += array[i + b *  batch_size]
#         batch_list.append(tmp)

#     return np.array(batch_list).astype(object)

# def unbatching_padding(array, elem_bits,batch_size ):
#     elem_bits = elem_bits + 4
#     res = []
#     mask = pow(2,elem_bits) - 1

#     for item in array:
#         item = int(item)
#         if item == 0 :
#             tmp = [0] * batch_size
#             res.extend(tmp)
#             continue
#         tmp = []
#         for i in range(batch_size):
#             num = item & mask
#             item >>= elem_bits
#             tmp.append(num)
#         tmp.reverse()
#         res += tmp
#     res = np.array(res)

#     return res

















































import numpy as np
from .aciq import ACIQ
from .complement import true2two, two2true

PLAIN_MODULUS = 1073872897
ALPHA = 600000.0 

precision = 5

# get aciq alpha
def get_alpha_r_max(plains,elements_bits, num_clients):
    list_min = np.min(plains)
    list_max = np.max(plains)
    list_size = len(plains)

    aciq  = ACIQ(elements_bits)
    alpha = aciq.get_alpha_gaus(list_min,list_max,list_size)
    r_max = alpha * num_clients
    return alpha, r_max

# quantize and padding  return two_complement
# def quantize_padding(value, alpha, quan_bits, num_clients):
#     # clipping
#     value = np.clip(value, -alpha, alpha)

#     # quantizing
#     sign = np.sign(value)
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * (pow(2, quan_bits - 1) - 1.0) / (alpha * num_clients)
#     value = unsigned_value * sign

#     # stochastic round
#     size = value.shape
#     value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)
    
#     # add padding bits and show with complement
#     padding_bits = int(np.ceil(np.log2(num_clients)))
#     res_val = []
#     for elem in value:
#         res_val.append(true2two(elem, padding_bits, quan_bits))
#     return res_val




# quantize and padding return
def quantize_padding(value, alpha, quan_bits, num_clients):
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
    
    # ATENÇÃO: Mudança aqui!
    # Substituindo o true2two pelo wrap-around do PLAIN_MODULUS
    res_val = []
    for elem in value:
        v = int(elem)
        # O módulo do Python faz o wrap-around perfeito para o HE: 
        # se v for positivo, fica igual. Se for negativo, vira PLAIN_MODULUS - |v|
        v = v % PLAIN_MODULUS 
        res_val.append(v)
        
    return res_val
# unquantize and return true value
# def unquantize_padding(value, alpha, quan_bits, num_clients):
#     value = np.array(value)
#     # stochastic round
#     size = value.shape
#     # value = np.floor(value + np.random.random(size)).astype(int)
#     value = value.astype(object)
#     padding_bits = int(np.ceil(np.log2(num_clients)))
#     # alpha *= pow(2,padding_bits)

#     # extract 2's complement to true value
#     res_value = []
#     for elem in value:
#         # if elem < 0:
#         #     raise ValueError("Overflow {}".format(elem))
#         #     # print("Overflow {}".format(elem))
#         #     # res_value.append(0)
#         #     # continue
#         if elem == 0:
#             res_value.append(elem)
#             continue
#         #elem &= ((1 << (quan_bits-1)) - 1)
#         elem = two2true(elem, padding_bits, quan_bits)
#         res_value.append(elem)
#     value = res_value

#     # unquantize
#     sign = np.sign(value)
#     unsigned_value = value * sign
#     unsigned_value = unsigned_value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
#     value = unsigned_value * sign 
#     return value


# unquantize_padding — remover o plain_modulus e o two2true
# usar diretamente a aritmética modular do plain_modulus


def unquantize_padding(value, alpha, quan_bits, num_clients):
    print(f"[DEBUG] unquantize_padding called, using PLAIN_MODULUS approach")
    value = np.array(value)
    value = value.astype(object)
    
    half = PLAIN_MODULUS // 2
    
    res_value = []
    for elem in value:
        v = int(elem) % PLAIN_MODULUS
        if v > half:
            v = v - PLAIN_MODULUS  # converte para valor negativo real
        res_value.append(v)
    
    value = np.array(res_value, dtype=object)
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
    value = unsigned_value * sign
    return value
def quan_no_compl(value, quan_bits, num_clients):
    alpha = 1.0
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
    value = value.astype(object)
    
    return value

def unquan_no_compl(value, quan_bits, num_clients):
    alpha = 1.0
    value = np.array(value)
    # stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int)
    value = value.astype(object)
 
    # unquantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
    value = unsigned_value * sign 
    return value  

def quantize_postive(value,quan_bits,num_clients):
    alpha = 1.0
    # clipping
    value = np.clip(value, -alpha, alpha)

    value = np.add(value,1.0)
    print(np.max(value))

    # quantizing
    value = value * (pow(2, quan_bits - 1) - 1.0) / (alpha * num_clients)

    # stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int)
    value = value.astype(object)

    return value

def unquantize_postive(value,quan_bits,num_clients,idx_weights):
    value = np.array(value)
    # stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int)
    value = value.astype(object)
    alpha = 1.0

    # unquantize
    value = value * (alpha * num_clients) / (pow(2, quan_bits - 1) - 1.0)
    #idx_weights = [0.14,0.35,0.45,0.5] 

    offset = np.sum(idx_weights)
    value = np.add(value, -1 * offset)
    return value

# quantize function
def quantize(plains,element_bits,num_clients):
    # alpha,r_max = get_alpha_r_max(plains,element_bits,num_clients)
    print("INICIANDO O QUANTIZE PADDING - Função")
    quantized = quantize_padding(plains,ALPHA,element_bits,num_clients)
    return np.array(quantized)

# unquantize function
# unquantize function
def unquantize(plains, element_bits, num_clients):
    print("INICIANDO O UNQUANTIZE PADDING")
    unquantized = unquantize_padding(plains, ALPHA, element_bits, num_clients)
    print("FINALIZADO O UNPADDING")
    
    # ATENÇÃO: Converter explicitamente para float antes de arredondar
    unquantized = np.array(unquantized, dtype=float) 
    
    return np.round(unquantized, precision)
# batch elems function
def batch_padding(array, max_bits, elem_bits,batch_size):
    elem_bits = elem_bits + 4
    array = array.tolist()
    #batch_size = max_bits // elem_bits
    if len(array) % batch_size != 0:
        pad_zero_nums = batch_size - len(array) % batch_size
        array += [0] * pad_zero_nums
  
    # how many batches is needed
    batch_nums = len(array) // batch_size
  
    # carry batches
    batch_list = []
    mod = pow(2, elem_bits)

    # for each batch
    for b in range(batch_nums):
        tmp = 0
        # for each elem
        for i in range(batch_size):
            tmp *= mod
            tmp += array[i + b *  batch_size]
        batch_list.append(tmp)

    return np.array(batch_list).astype(object)

def unbatching_padding(array, elem_bits,batch_size ):
    elem_bits = elem_bits + 4
    res = []
    mask = pow(2,elem_bits) - 1

    for item in array:
        item = int(item)
        if item == 0 :
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
    res = np.array(res)

    return res