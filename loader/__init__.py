from loader.CT_loader import CTLoader
from loader.GAL_loader import GALoader
from loader.testloader import TestLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "ct": CTLoader,
        "CT": CTLoader
    }[name]
