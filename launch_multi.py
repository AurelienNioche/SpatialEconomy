from multiprocessing import Pool
import argparse
import pickle
from c_writer_main_wisdom_of_crowds import SimulationRunner, BackUp


def launch(parameters):

    results = SimulationRunner.launch_economy(parameters=parameters, graphics=None)
    BackUp.save_data(results, parameters, graphics=None)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('parameters_list_name', type=str,
                        help='A name of pickle file for parameters is required!')
    parser.add_argument('number_of_processes', type=int,
                        help='A name of authorized processes is required!')

    args = parser.parse_args()

    parameters_list = pickle.load(open(name=args.parameters_list_name, mode='rb'))

    pool = Pool(processes=args.number_of_processes)

    pool.map(SimulationRunner.main_runner, parameters_list)


if __name__ == '__main__':

    main()
