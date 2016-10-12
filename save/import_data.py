import pickle
from arborescence.arborescence import Folders


def import_data(suffix, data_folder=Folders.folders["data"]):

    parameters = pickle.load(open("{}/parameters/parameters_{}.p".format(data_folder, suffix), mode="rb"))
    # print(parameters)

    direct_exchange = pickle.load(
        open("{}/exchanges/direct_exchanges_{}.p".format(data_folder, suffix), mode="rb"))
    indirect_exchange = pickle.load(
        open("{}/exchanges/indirect_exchanges_{}.p".format(data_folder, suffix), mode="rb"))

    return parameters, direct_exchange, indirect_exchange
