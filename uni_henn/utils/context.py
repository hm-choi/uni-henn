from seal import *
from uni_henn.constants import *

class Context:
    def _get_params(self, poly_modulus_degree, coeff_modulus):
        """Get default parameters"""
        params = EncryptionParameters(scheme_type.ckks)
        params.set_poly_modulus_degree(poly_modulus_degree)
        params.set_coeff_modulus(
            CoeffModulus.Create(poly_modulus_degree, coeff_modulus)
        )
        return params
        
    def _generate_keys(self, context):
        keygen = KeyGenerator(context)
        self.public_key = keygen.create_public_key()
        self.secret_key = keygen.secret_key()
        self.galois_key = keygen.create_galois_keys()
        self.relin_keys = keygen.create_relin_keys()
    
    def __init__(self, N = NUMBER_OF_SLOTS, depth = DEPTH, LogQ = FRACTION_SCALE, LogP = INTEGER_SCALE + FRACTION_SCALE, scale = 0):
        """Initialize for context"""
        coeff_modulus = [LogP] + [LogQ] * depth + [LogP]
        context = SEALContext(self._get_params(N * 2, coeff_modulus))

        self.number_of_slots = N
        self.depth = depth
        if scale == 0:
            scale = 2**LogQ
        self.scale = scale

        self._generate_keys(context)

        self.encoder = CKKSEncoder(context)
        self.encryptor = Encryptor(context, self.public_key)
        self.evaluator = Evaluator(context)
        self.decryptor = Decryptor(context, self.secret_key)

        # self.public_key.save('key/public_key')
        # self.secret_key.save('key/secret_key')
        # self.galois_key.save('key/galois_key')
        # self.relin_keys.save('key/relin_keys')