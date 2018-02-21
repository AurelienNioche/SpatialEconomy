# -*- coding: utf-8 -*-
from multiprocessing import Pool
import argparse
import pickle
from eco.c_economy import SimulationRunner
from save.save_eco import BackUp
from arborescence.arborescence import Folders
from __old__.merge_db import merge_db


def launch(parameters):

    results = SimulationRunner.multi_launch_economy(parameters)
    back_up_db_name = BackUp.save_data(results, parameters, graphics=None)
    return back_up_db_name


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('parameters_list_name', type=str,
                        help='A name of pickle file for parameters is required!')
    parser.add_argument('number_of_processes', type=int,
                        help='A name of authorized processes is required!')

    args = parser.parse_args()

    parameters_list = pickle.load(open("{}/{}".format(
        Folders.folders["parameters"],
        args.parameters_list_name), mode='rb'))

    # Launch the process

    pool = Pool(processes=args.number_of_processes)
    back_up_db_names = pool.map(launch, parameters_list)
    merge_db(db_folder=Folders.folders["data"], db_to_merge=back_up_db_names,
             new_db_name="data_{}".format(args.parameters_list_name.split(".")[0]))


if __name__ == '__main__':

    main()
