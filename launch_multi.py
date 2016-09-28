# -*- coding: utf-8 -*-
from multiprocessing import Pool
import argparse
import pickle
from eco.c_economy import SimulationRunner
from save.save_eco import BackUp


def launch(parameters):

    results = SimulationRunner.multi_launch_economy(parameters)
    BackUp.save_data(results, parameters, graphics=None)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('parameters_list_name', type=str,
                        help='A name of pickle file for parameters is required!')
    parser.add_argument('number_of_processes', type=int,
                        help='A name of authorized processes is required!')

    args = parser.parse_args()

    parameters_list = pickle.load(open(args.parameters_list_name, mode='rb'))

    # Launch the process

    pool = Pool(processes=args.number_of_processes)
    pool.map(launch, parameters_list)


if __name__ == '__main__':

    main()
