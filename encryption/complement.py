
# true value trans to 2's complement
# def true2two(value, padding_bits, quan_bits):
#     if value >= 0:
#         binary = format(value,'0{}b'.format(quan_bits))
#         return int(binary.zfill(padding_bits + quan_bits),2)
#     else:
#         binary = format(abs(value),'0{}b'.format(quan_bits))
#         inversed = ''.join('1' if bit == '0' else '0' for bit in binary)
#         complement = int(bin(int(inversed,2)+1)[2:].zfill(padding_bits + quan_bits),2)
#         return complement

# 2's complement trans to true value
# def two2true(value, padding_bits, quan_bits):
#     #value = int(value )
#     mod = pow(2,quan_bits)
#     #value = value % mod
#     if value > mod:
#         value %= mod
#     value = bin(value)[2:].zfill(quan_bits)

#     if value[0] == '0':
#         return int(value,2)
#     elif value[0] == '1':
#         inversed = ''.join('1' if bit == '0' else '0' for bit in value)
#         value = int(inversed,2) + 1
#         if value >= pow(2, quan_bits) - 1:
#             value -= pow(2, quan_bits)
#         return (-1*value)
#     else:
#         raise ValueError("Overflow {}".format(value))

#  ESse da certo
# def two2true(value, padding_bits, quan_bits):
#     total_bits = quan_bits  # quan_bits já é total_bits quando chamado do unquantize_padding
#     mod = pow(2, total_bits)
#     value = int(value) % mod
#     value_bin = bin(value)[2:].zfill(total_bits)[-total_bits:]
    
#     if value_bin[0] == '0':
#         return int(value_bin, 2)
#     else:
#         inversed = ''.join('1' if bit == '0' else '0' for bit in value_bin)
#         result = -(int(inversed, 2) + 1)
#         return result
    
    
# def true2two(value, padding_bits, quan_bits):
#     total_bits = padding_bits + quan_bits  # 26
#     if value >= 0:
#         return value  # já é positivo, BFV cuida do resto
#     else:
#         return (1 << total_bits) + value  # complemento de dois em 26 bits



# Paillier

def true2two(value, padding_bits, quan_bits):
    total_bits = padding_bits + quan_bits 
    if value >= 0:
        return value  
    else:
        return (1 << total_bits) + value 
        
def two2true(value, padding_bits, quan_bits):
    print("[DEBUG] Esse ta fora do /cliente, so pra saber onde ele ta chamando")
    print("[DEBUG] two2true called with value={}, padding_bits={}, quan_bits={}".format(value, padding_bits, quan_bits))
    print(f"two2true called with value={value}, padding_bits={padding_bits}, quan_bits={quan_bits}, total_bits={padding_bits + quan_bits}, value % mod={int(value) % pow(2, padding_bits + quan_bits)}")
    # O total_bits na volta PRECISA ser a soma, igual na ida.
    total_bits = padding_bits + quan_bits  
    
    mod = pow(2, total_bits)
    value = int(value) % mod
    
    # Extrai a representação binária exata do bloco
    value_bin = bin(value)[2:].zfill(total_bits)[-total_bits:]
    
    # Verifica o bit mais à esquerda (sinal)
    if value_bin[0] == '0':
        return int(value_bin, 2)
    else:
        # Reverte o complemento de 2
        inversed = ''.join('1' if bit == '0' else '0' for bit in value_bin)
        result = -(int(inversed, 2) + 1)
        return result