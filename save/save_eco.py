from save.converter import write
from pickle import dump
from tqdm import tqdm_gui
from collections import OrderedDict
from arborescence.arborescence import Folders
from save.database import Database
from os import path



class BackUp(object):

    @classmethod
    def save_data(cls, results, parameters, graphics=1):

        folders = Folders.folders

        try:
            date = parameters["date"]

        except KeyError:
            from datetime import date
            date = str(date.today()).replace("-", "_")

        try:
            idx = parameters["idx"]
        except KeyError:
            idx = 0
            while path.exists("{folder}/{date}_idx{idx}.db".format(folder=folders["data"], date=date, idx=idx)):
                idx += 1

        tqdm_gui.write("\nSaving data...")

        saving_name = "{date}_idx{idx}".format(date=date, idx=idx)

        db = Database(folder=folders["data"], database_name=saving_name)

        cls.save_parameters(db=db, idx=idx, parameters=parameters)
        cls.save_exchanges(db=db, idx=idx,
                           direct_exchanges=results["direct_choices"],
                           indirect_exchanges=results["indirect_choices"])

        if graphics:

            cls.save_graphic_data(results=results, saving_name=saving_name, folders=folders)

        tqdm_gui.write("\nData saved.")

        return saving_name

    @classmethod
    def save_parameters(cls, db, idx, parameters):

        columns = OrderedDict([
            ("vision_area", "INTEGER"),
            ("movement_area", "INTEGER"),
            ("stride", "INTEGER"),
            ("width", "INTEGER"),
            ("height", "INTEGER"),
            ("x0", "INTEGER"),
            ("x1", "INTEGER"),
            ("x2", "INTEGER"),
            ("alpha", "FLOAT"),
            ("tau", "FLOAT"),
            ("t_max", "INTEGER")
        ])

        parameters_table = "parameters_{}".format(idx)

        db.create_table(table_name=parameters_table,
                        columns=columns)

        param_to_save = [[]]
        for column_name in columns.keys():
            param_to_save[0].append(parameters[column_name])

        db.write_n_rows(table_name=parameters_table,
                        columns=columns,
                        array_like=param_to_save)

    @classmethod
    def save_exchanges(cls, db, idx, direct_exchanges, indirect_exchanges):

        columns = OrderedDict([
            ("x0", "FLOAT"),
            ("x1", "FLOAT"),
            ("x2", "FLOAT")
        ])

        direct_exchanges_table = "direct_exchanges_{}".format(idx)
        indirect_exchanges_table = "indirect_exchanges_{}".format(idx)

        db.create_table(table_name=direct_exchanges_table,
                        columns=columns)

        db.create_table(table_name=indirect_exchanges_table,
                        columns=columns)

        db.write_n_rows(table_name=direct_exchanges_table,
                        columns=columns,
                        array_like=direct_exchanges)

        db.write_n_rows(table_name=indirect_exchanges_table,
                        columns=columns,
                        array_like=indirect_exchanges)

    @classmethod
    def save_graphic_data(cls, results, saving_name, folders):

        matrix_list = results["matrix_list"]
        list_map = results["list_map"]

        for i in range(len(matrix_list)):
            write(matrix_list[i], table_name="exchange_{}".format(i),
                  database_name='{}/array_exchanges_{}'.format(folders["data"], saving_name),
                  descr="{}/3".format(i + 1))

        dump(list_map,
             open("{}/position_map{}.p".format(folders["data"], saving_name), mode='wb'))