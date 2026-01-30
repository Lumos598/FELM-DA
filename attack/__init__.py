from .gaussian import GaussianAttack
from .converse import ConverseAttack
from .random import RandomAttack
from .little import LittleAttack
from .hidden import HiddenAttack

attacks = {
    'gaussian': GaussianAttack,
    'converse': ConverseAttack,
    'random': RandomAttack,
    'hidden': HiddenAttack,
    'little': LittleAttack,
}