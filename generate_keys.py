import encryption.quantize
import tenseal as ts
import os
import pickle
import numpy as np
# from .client.encryption.paillier import PaillierCipher

# from client.encryption import PaillierCipher
# from .client.encryption.paillier import PaillierCipher
from encryption.paillier import PaillierCipher


def context_bfv(poly_modulus_degree=8192,plain_modulus=1073872897,key_type='none'):
    """
    This function is used to create the context of the homomorphic encryption:
    it is used to create the keys and the parameters of the encryption scheme (BFV).

    :return: the context of the homomorphic encryption
    """
    cont = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[50,50,50,50],
        plain_modulus= 1073872897,
    )
    
    if key_type== 'all':
        cont.generate_galois_keys()
        cont.generate_relin_keys()
    elif key_type == 'galois':
        cont.generate_galois_keys()
    elif key_type == 'relin':
        cont.generate_relin_keys()
    else:   
        pass
           # You can create the Galois keys by calling generate_galois_keys

    # You can create the Galois keys by calling generate_galois_keys
    # cont.generate_relin_keys()
    # cont.global_scale = 2 ** 40  # global_scale: the scaling factor, here set to 2**40 (same that pow(2, 40))
    return cont

# def context_ckks(poly_modulus_degree=8192,coeff_mod_bit_sizes=[60, 40, 40, 60],global_scale=2**40,key_type='galois'):
#     """
#     This function is used to create the context of the homomorphic encryption:
#     it is used to create the keys and the parameters of the encryption scheme (CKKS).

#     :return: the context of the homomorphic encryption
#     """
#     cont = ts.context(
#         ts.SCHEME_TYPE.CKKS,
#         poly_modulus_degree=poly_modulus_degree if poly_modulus_degree != None else 8192,
#         coeff_mod_bit_sizes=coeff_mod_bit_sizes if coeff_mod_bit_sizes != None else [60, 40, 40, 60]
#         # This means that the coefficient modulus will contain 4 primes of 60 bits, 40 bits, 40 bits, and 60 bits.
#         # coeff_mod_bit_sizes=[40, 21, 21, 40]
#     )
#     if key_type== 'all':
#         cont.generate_galois_keys()
#         cont.generate_relin_keys()
#     elif key_type == 'galois':
#         cont.generate_galois_keys()
#     elif key_type == 'relin':
#         cont.generate_relin_keys()
#     else:   
#         pass
#     # You can create the Galois keys by calling generate_galois_keys
#     # cont.generate_relin_keys()
#     cont.global_scale = global_scale if global_scale != None else 2 ** 40
#       # global_scale: the scaling factor, here set to 2**40 (same that pow(2, 40))
#     return cont
def context_ckks(poly_modulus_degree=8192,coeff_mod_bit_sizes=[60, 40, 40, 60],global_scale=2**40,key_type='galois'):
    """
    This function is used to create the context of the homomorphic encryption:
    it is used to create the keys and the parameters of the encryption scheme (CKKS).

    :return: the context of the homomorphic encryption
    """
    cont = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=  8192,
        coeff_mod_bit_sizes= [30,60,30]
   
    )
    cont.global_scale = 2**60
    return cont
def read_query(file_path):
    """
    This function is used to read a pickle file.

    :param file_path: the path of the file to read
    :return: the query and the context
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            """
            # pickle.load(f)  # load to read file object

            file_str = f.read()
            client_query1 = pickle.loads(file_str)  # loads to read str class
            """
            query_str = pickle.load(file)

        contexte = query_str["context"]  # ts.context_from(query["contexte"])
        del query_str["context"]
        return query_str, contexte

    else:
        print("The file doesn't exist")

def write_query(file_path, client_query):
    """
    This function is used to write a pickle file.

    :param file_path: the path of the file to write

    :param client_query: the query to write
    """
    with open(file_path, 'wb') as file:  # 'ab' to add existing file
        encode_str = pickle.dumps(client_query)
        file.write(encode_str)

def combo_keys(type_key='ckks', client_path="secret.pkl", server_path="server_key.pkl",args={}):
    """
    To create the public/private keys combination
    args:
        client_path: path to save the secret key (str)
        server_path: path to save the server public key (str)
    """

    poly_modulus_degree=None if args.get("poly_modulus_degree") == None else args["poly_modulus_degree"]
    plain_modulus=None if args.get("plain_modulus") == None else args["plain_modulus"]
    key_type = None if args.get("key_type" ) == None else args["key_type"]
    coeff_mod_bit_sizes=None if args.get("coeff_mod_bit_sizes") == None else args["coeff_mod_bit_sizes"]
    global_scale=None if args.get("global_scale") == None else args["global_scale"]
   
    print(f"poly_modulus_degree {poly_modulus_degree}")
    print(f"plain_modulus {plain_modulus}")
    print(f"key_type {key_type}")
    print(f"coeff_mod_bit_sizes {coeff_mod_bit_sizes}")
    print(f"global_scale {global_scale}")
    n_length=None  if args.get("n_length") == None else args["n_length"]
    print(f'Generating {args} keys...')



    if type_key == 'ckks':
        context_client = context_ckks(poly_modulus_degree=poly_modulus_degree,coeff_mod_bit_sizes=coeff_mod_bit_sizes,global_scale=global_scale,key_type=key_type)
    else:
        context_client = context_bfv(plain_modulus=plain_modulus,poly_modulus_degree=poly_modulus_degree,key_type=key_type)
    safe_scale = int(global_scale) if global_scale is not None else 0
    safe_key_type = key_type if key_type is not None else 'None'
    if type_key == "ckks":
        client_key_name = "nvnlc2.pkl"
        server_key_name= "serverc2.pkl"
        # server_key_name = f"{type_key}_{poly_modulus_degree}_{plain_modulus}_{coeff_mod_bit_sizes}_{safe_scale}_{safe_key_type}_{server_path}"
        # client_key_name = f"{type_key}_{poly_modulus_degree}_{plain_modulus}_{coeff_mod_bit_sizes}_{safe_scale}_{safe_key_type}_{client_path}"
    else:
        server_key_name=f"{type_key}_{server_path}"
        client_key_name=f"{type_key}_{client_path}"
    # server_key_name = f"{type_key}_{poly_modulus_degree}_{plain_modulus}_{coeff_mod_bit_sizes}_{int(global_scale)}_{key_type if key_type != None else 'None'}_{server_path}"
    # client_key_name = f"{type_key}_{poly_modulus_degree}_{plain_modulus}_{coeff_mod_bit_sizes}_{global_scale}_{key_type}_{client_path}"
    write_query(f'context/{client_key_name}', {"context": context_client.serialize(save_secret_key=True)})
    write_query(f'context/{server_key_name}', {"context": context_client.serialize()})

    _, context_client = read_query(f'context/{client_key_name}')
    _, context_server = read_query(f'context/{server_key_name}')

    context_client = ts.context_from(context_client)
    context_server = ts.context_from(context_server)
    print("Is the client context private?", ("Yes" if context_client.is_private() else "No"))
    print("Is the server context private?", ("Yes" if context_server.is_private() else "No"))


def Paillier_keys(n_length=2048):
    paillier = PaillierCipher()
    paillier.generate_key(n_length=n_length if n_length != None else 2048)
    print(n_length)
    with open("./context/paillier.pkl", "wb") as file:
        pickle.dump(paillier, file)

def main():
    path = "context"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # combo_keys('ckks')
    combo_keys('ckks')
    # Paillier_keys()

if __name__ == "__main__":
    main()