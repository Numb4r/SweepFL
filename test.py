# import tenseal as ts
# from typing import Any

# coeff = [30, 20,20, 30]  # total = 122 bits (ainda alto)
# print(sum(coeff))
# cont: Any = ts.context( # type: ignore
#         ts.SCHEME_TYPE.CKKS,
#         poly_modulus_degree= 4096,
#         # This means that the coefficient modulus will contain 4 primes of 60 bits, 40 bits, 40 bits, and 60 bits.
#         coeff_mod_bit_sizes= coeff
#         # coeff_mod_bit_sizes=[40, 21, 21, 40]
#     ) 
# cont.global_scale = 2**(min(coeff)-2)

# # Bits maximo sum(coeff)
# # | poly_modulus_degree | Máximo total de bits |
# # | ------------------- | -------------------- |
# # | **4096**            | **109 bits**         |
# # | **8192**            | **218 bits**         |
# # | **16384**           | **438 bits**         |
import pickle 
# with open(f'context/paillier.pkl', 'rb') as file:
import tenseal as ts 
#     context = pickle.load(file)    
with open(f"./context/ckks_8192_-1_[30, 60, 30]_1152921504606846976_None_secret.pkl",'rb') as file:
    # context = pickle.load(file)
    secret = pickle.load(file) 
#     # print(context)
    context = ts.context_from(secret['context'])

a = [0.1,0.2,0.3]
b = [0.1,0.2,0.3]
ac = ts.ckks_vector(context,a).serialize()
# print(ac)
bc = ts.ckks_vector(context,b).serialize()
# cc = ac+bc 
# print(cc.decrypt())

with open(f"./context/ckks_8192_-1_[30, 60, 30]_1152921504606846976_None_secret.pkl",'rb') as file:
    # context = pickle.load(file)
    secret = pickle.load(file) 
#     # print(context)
    server = ts.context_from(secret['context'])
    
ass = ts.ckks_vector_from(server,ac)
bss = ts.ckks_vector_from(server,bc)
css = ass+bss 
cs = css.serialize()

c = ts.ckks_vector_from(context,cs).decrypt()
print(c)