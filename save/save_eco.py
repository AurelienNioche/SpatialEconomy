from save.converter import write
from pickle import dump
from tqdm import tqdm_gui
from arborescence.arborescence import Folders


class BackUp(object):

    @classmethod
    def save_data(cls, results, parameters, graphics=1):

        folders = Folders.folders

        tqdm_gui.write("\nSaving data...")

        saving_name = "{date}_idx{idx}".format(date=parameters["date"], idx=parameters["idx"])

        if graphics:

            matrix_list = results["matrix_list"]
            list_map = results["list_map"]

            for i in range(len(matrix_list)):
                write(matrix_list[i], table_name="exchange_{}".format(i),
                      database_name='{}/array_exchanges{}'.format(folders["data"], saving_name),
                      descr="{}/3".format(i + 1))

            dump(list_map,
                 open("{}/position_map{}.p".format(folders["data"], saving_name), mode='wb'))

        direct_exchanges = results["direct_choices"]
        indirect_exchanges = results["indirect_choices"]

        dump(direct_exchanges,
             open("{}/direct_exchanges_{}.p".format(folders["data"], saving_name), mode='wb'))
        dump(indirect_exchanges,
             open("{}/indirect_exchanges_{}.p".format(folders["data"], saving_name), mode='wb'))

        dump(parameters,
             open("{}/parameters_{}.p".format(folders["data"], saving_name), mode='wb'))

        tqdm_gui.write("\nData saved...")
