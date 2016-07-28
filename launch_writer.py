import numpy as np
from eco.c_economy import SimulationRunner
from save.save_eco import BackUp


def simple_main():

    '''
    Simplest program use
    :return: None
    '''

    parameters = \
        {
            "workforce": np.array([50, 50, 50], dtype=int),
            "alpha": 0.1,  # Set the coefficient learning
            "tau": 0.01,  # Set the softmax parameter.
            "t_max": 1000,  # Set the number of time units the simulation will run
            "stride": 1,  # by each agent at each round
            "vision": 5,  # Set the importance of other agents'results in
            "area": 5,  # front of an individual res
            "map_limits": {"width": 30, "height": 30},
            "idx": 0

        }

    results = SimulationRunner.launch_economy(parameters=parameters, graphics=0)

    BackUp.save_data(results, parameters, graphics=1)


if __name__ == "__main__":

    simple_main()
