from loader.CT_loader import CTLoader


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "ct": CTLoader,
        "CT": CTLoader
    }[name]
