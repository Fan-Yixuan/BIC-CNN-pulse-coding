from yacs.config import CfgNode

_C = CfgNode()

_C.NAME = ''
_C.SEED = 0
_C.MODE = ''

_C.DIR = CfgNode()
_C.DIR.RESULT = './result'
_C.DIR.DATA = './'

_C.MODEL = CfgNode()
_C.MODEL.WINDOW = 32
_C.MODEL.TH = 0.3
_C.MODEL.LEN = 0.5
_C.MODEL.DECAY = 0.7

_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 30
_C.TRAIN.LR = 5e-3
_C.TRAIN.BS = 64
_C.TRAIN.ONE_INIT = False

_C.TEST = CfgNode()
_C.TEST.BS = 1000

cfg = _C
