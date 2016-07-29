from save.converter import write
from pickle import dump
from tqdm import tqdm_gui
from os import path, mkdir


class BackUp(object):

    @classmethod
    def save_data(cls, results, parameters, graphics=1):

        cls.create_folders()

        tqdm_gui.write("\nSaving data...")

        saving_name = "{date}_idx{idx}".format(date=parameters["date"], idx=parameters["idx"])

        if graphics:

            matrix_list = results["matrix_list"]
            list_map = results["list_map"]

            for i in range(len(matrix_list)):
                write(matrix_list[i], table_name="exchange_{}".format(i),
                      database_name='array_exchanges{}'.format(saving_name), descr="{}/3".format(i + 1))

            dump(list_map, open("../../data/positions/position_map{}.p".format(saving_name), mode='wb'))

        direct_exchanges = results["direct_choices"]
        indirect_exchanges = results["indirect_choices"]

        dump(direct_exchanges, open("../../data/exchanges/direct_exchanges_{}.p".format(saving_name), mode='wb'))
        dump(indirect_exchanges,
                    open("../../data/exchanges/indirect_exchanges_{}.p".format(saving_name), mode='wb'))

        dump(parameters, open("../../data/parameters/parameters_{}.p".format(saving_name), mode='wb'))

        tqdm_gui.write("\nData saved...")

    @classmethod
    def create_folders(cls):

        folders = ["../../data", "../../data/parameters", "../../data/exchanges", "../../data/positions"]
        for i in folders:

            if not path.exists(i):

                mkdir(i)
