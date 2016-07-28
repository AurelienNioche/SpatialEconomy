import pickle


def import_data(suffix):

    parameters = pickle.load(open("../data/parameters/parameters_{}.p".format(suffix), mode="rb"))
    print(parameters)

    direct_exchange = pickle.load(open("../data/exchanges/direct_exchanges_{}.p".format(suffix), mode="rb"))
    indirect_exchange = pickle.load(open("../data/exchanges/ indirect_exchanges_{}.p".format(suffix), mode="rb"))

    return parameters, direct_exchange, indirect_exchange
