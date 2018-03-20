import yaml
from easydict import EasyDict

_conf = None


def read_conf(pth):
    return EasyDict(yaml.load(open(pth)))


def load_global_conf(pth):
    global _conf
    _conf = read_conf(pth)


def get_global_conf():
    global _conf
    return _conf
