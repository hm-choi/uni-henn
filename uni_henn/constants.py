import torch
from torchvision import transforms

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMBER_OF_SLOTS = 8192
POLY_MODULUS_DEGREE = NUMBER_OF_SLOTS * 2
INTEGER_SCALE = 8
FRACTION_SCALE = 32
SCALE = 2**32
DEPTH = 11

COEFF_MODULUS = [INTEGER_SCALE + FRACTION_SCALE] + \
                [FRACTION_SCALE] * DEPTH + \
                [INTEGER_SCALE + FRACTION_SCALE]

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])