import pickle
from arborescence.arborescence import Folders


def import_data(suffix):

    parameters = pickle.load(open("{}/parameters_{}.p".format(Folders.folders["data"], suffix), mode="rb"))
    # print(parameters)

    direct_exchange = pickle.load(
        open("{}/direct_exchanges_{}.p".format(Folders.folders["data"], suffix), mode="rb"))
    indirect_exchange = pickle.load(
        open("{}/indirect_exchanges_{}.p".format(Folders.folders["data"], suffix), mode="rb"))

    return parameters, direct_exchange, indirect_exchange


def import_parameters(suffix):

    parameters = pickle.load(
        open("{}/parameters_{}.p".format(Folders.folders["data"], suffix), mode="rb"))
    # print(parameters)
    return parameters
