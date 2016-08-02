import pickle


def import_data(suffix):

    parameters = pickle.load(open("../data/parameters/parameters_{}.p".format(suffix), mode="rb"))
    # print(parameters)

    direct_exchange = pickle.load(open("../data/exchanges/direct_exchanges_{}.p".format(suffix), mode="rb"))
    indirect_exchange = pickle.load(open("../data/exchanges/indirect_exchanges_{}.p".format(suffix), mode="rb"))

    return parameters, direct_exchange, indirect_exchange


def import_parameters(suffix):

    parameters = pickle.load(open("../data/parameters/parameters_{}.p".format(suffix), mode="rb"))
    # print(parameters)
    return parameters


def import_suffixes(session_suffix):

    suffixes = pickle.load(open("../data/session/session_{}.p".format(session_suffix), mode='rb'))
    return suffixes

