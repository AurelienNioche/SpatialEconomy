import numpy as np
from writer_main_optimized import SimulationRunner, BackUp


def simple_main():

    '''
    Simplest program use
    :return: None
    '''

    parameters = \
        {
            "workforce": np.array([10, 10, 10], dtype=int),
            "alpha": 0.3,  # Set the coefficient learning
            "tau": 0.03,  # Set the softmax parameter.
            "t_max": 1000,  # Set the number of time units the simulation will run
            "stride": 1,  # by each agent at each round
            "epsilon": 0.3,
            "vision": 10,  # Set the importance of other agents'results in
            "area": 10,  # front of an individual res
            "map_limits": {"width": 10, "height": 10},

        }

    results = SimulationRunner.main_runner(parameters=parameters)

    BackUp.save_data(results, parameters, graphics=0)

if __name__ == "__main__":

    simple_main()
