import argparse 
from multiprocessing import Pool
import pickle 
from writer_main_wisdom_of_crowds import Economy, SimulationRunner

parser = argparse.ArgumentParser() 
parser.add_argument('parameters', type=str, 
                    help='A name of pickle file for parameters is required !\
                          Feed me ! ') 
 
args = parser.parse_args().parameters

print("Loading {}".format(args))

data_list = pickle.load(open("../data/{}.p".format(args), mode='rb'))


p = Pool()

result = p.map(SimulationRunner.main_runner, data_list)


pickle.dump(result, open("../data/figures/exchanges_slice_nb_{}.p".format(
            data_list[0]['idx']), mode='wb'))
