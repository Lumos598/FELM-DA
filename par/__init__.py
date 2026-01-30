from .qc import QC
from .avg import Average
from .felm_da import FELM_DA
from .bristle import Bristle
from .balance import Balance
from .d_fltrust import DFLTrust
from .scclip import SCClip
from .d_krum import DKrum
from .d_tofi import DToFi
from .d_median import DMedian

pars = {
    "qc": QC,
    "avg": Average,
    "felm_da":FELM_DA,
    "bristle": Bristle,
    "balance": Balance,
    "fltrust": DFLTrust,
    "scclip": SCClip,
    "krum": DKrum,
    "tofi": DToFi,
    "median": DMedian
}