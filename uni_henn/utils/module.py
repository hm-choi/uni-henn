from seal import *
from uni_henn.constants import *

class Context:
    def _get_params(self):
        """Get default parameters"""
        params = EncryptionParameters(scheme_type.ckks)
        params.set_poly_modulus_degree(POLY_MODULUS_DEGREE)
        params.set_coeff_modulus(
            CoeffModulus.Create(POLY_MODULUS_DEGREE, COEFF_MODULUS)
        )
        return params
        
    def _generate_keys(self, context):
        keygen = KeyGenerator(context)
        self.public_key = keygen.create_public_key()
        self.secret_key = keygen.secret_key()
        self.galois_key = keygen.create_galois_keys()
        self.relin_keys = keygen.create_relin_keys()
    
    def __init__(self):
        """Initialize for context"""
        context = SEALContext(self._get_params())
        self.slot_count = NUMBER_OF_SLOTS
        
        self._generate_keys(context)

        self.encoder = CKKSEncoder(context)
        self.encryptor = Encryptor(context, self.public_key)
        self.evaluator = Evaluator(context)
        self.decryptor = Decryptor(context, self.secret_key)

        # self.public_key.save('key/public_key')
        # self.secret_key.save('key/secret_key')
        # self.galois_key.save('key/galois_key')
        # self.relin_keys.save('key/relin_keys')