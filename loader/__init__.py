from loader.CT_loader import CTLoader
from loader.GAL_loader import GALoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "ct": CTLoader,
        "CT": CTLoader
    }[name]
