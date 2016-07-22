import numpy as np
from writer_main_wisdom_of_crowds import SimulationRunner, BackUp


def simple_main():

    '''
    Simplest program use
    :return: None
    '''

    parameters = \
        {
            "workforce": np.array([10, 10, 10], dtype=int),
            "alpha": 0.4,  # Set the coefficient learning
            "tau": 0.02,  # Set the softmax parameter.
            "t_max": 50000,  # Set the number of time units the simulation will run
            "stride": 1,  # by each agent at each round
            "vision": 5,  # Set the importance of other agents'results in
            "area": 5,  # front of an individual res
            "map_limits": {"width": 10, "height": 10},

        }

    results = SimulationRunner.main_runner(parameters=parameters)

    BackUp.save_data(results, parameters, graphics=0)

if __name__ == "__main__":

    simple_main()
