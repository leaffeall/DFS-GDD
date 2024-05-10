from utils.registry import LOSSFUNC
from .cross_entropy_loss import CrossEntropyLoss
from .consistency_loss import ConsistencyCos
# from .capsule_loss import CapsuleLoss
from .bce_loss import BCELoss
from .bi_level_CE import BiCE
# from .am_softmax import AMSoftmaxLoss
# from .am_softmax import AMSoftmax_OHEM
from .contrastive_regularization import ContrastiveLoss
from .l1_loss import L1Loss
# from .id_loss import IDLoss
# from .vgg_loss import VGGLoss
from .daw_fdd import DAW
from .bal_loss import LDAMLoss
from .mi_loss import MIloss
