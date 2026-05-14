import os
import pickle
import numpy as np
from encryption.paillier import PaillierCipher


def context_bfv(poly_modulus_degree=8192, plain_modulus=1073872897, key_type='none'):
    import tenseal as ts
    cont = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[50, 50, 50, 50],
        plain_modulus=1073872897,
    )
    if key_type == 'all':
        cont.generate_galois_keys()
        cont.generate_relin_keys()
    elif key_type == 'galois':
        cont.generate_galois_keys()
    elif key_type == 'relin':
        cont.generate_relin_keys()
    return cont


def read_query(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            query_str = pickle.load(file)
        contexte = query_str["context"]
        del query_str["context"]
        return query_str, contexte
    else:
        print("The file doesn't exist")


def write_query(file_path, client_query):
    with open(file_path, 'wb') as file:
        encode_str = pickle.dumps(client_query)
        file.write(encode_str)


def combo_keys(type_key='ckks', client_path="secret.pkl", server_path="server_key.pkl", args={}):
    """Generate and persist HE keys for the given scheme.

    For CKKS, delegates to the backend selected by args['ckks_backend']
    (falls back to the CKKS_BACKEND env var, then to 'tenseal').
    BFV always uses TenSEAL regardless of the CKKS backend setting.
    """
    from encryption.ckks_backend import get_backend

    context_dir = os.environ.get("CONTEXT_DIR", "context")
    os.makedirs(context_dir, exist_ok=True)

    print(f'Generating {type_key} keys with args: {args}')

    if type_key == 'ckks':
        backend_name = args.get("ckks_backend") or os.environ.get("CKKS_BACKEND", "tenseal")
        backend = get_backend(backend_name)
        backend.generate_and_save(args, context_dir)
        print(f"CKKS keys generated via backend '{backend_name}'.")

    else:  # BFV — always TenSEAL
        context_client = context_bfv(
            plain_modulus=args.get("plain_modulus"),
            poly_modulus_degree=args.get("poly_modulus_degree"),
            key_type=args.get("key_type"),
        )
        client_key_name = f"bfv_{client_path}"
        server_key_name = f"bfv_{server_path}"
        write_query(f'{context_dir}/{client_key_name}', {"context": context_client.serialize(save_secret_key=True)})
        write_query(f'{context_dir}/{server_key_name}', {"context": context_client.serialize()})
        print(f"BFV keys written: {client_key_name}, {server_key_name}")


def Paillier_keys(n_length=2048):
    context_dir = os.environ.get("CONTEXT_DIR", "context")
    os.makedirs(context_dir, exist_ok=True)
    paillier = PaillierCipher()
    paillier.generate_key(n_length=n_length if n_length is not None else 2048)
    with open(f"{context_dir}/paillier.pkl", "wb") as file:
        pickle.dump(paillier, file)


def main():
    combo_keys('ckks')


if __name__ == "__main__":
    main()
