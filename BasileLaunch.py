import numpy as np
from writer_main_wisdom_of_crowds import SimulationRunner, BackUp

# kikoo basile

def simple_main():

    '''
    Simplest program use
    :return: None
    '''
   
    parameters = \
        {
            "workforce": np.array([35, 35, 35], dtype=int),
            "alpha": 0.4,  # Set the coefficient learning
            "tau": 0.02,  # Set the softmax parameter.
            "t_max": 1200,  # Set the number of time units the simulation will run
            "stride": 1,  # by each agent at each round
            "vision": 6,  # Set the importance of other agents'results in
            "area": 10,  # front of an individual res
            "map_limits": {"width": 20, "height": 20},

        }

    results = SimulationRunner.main_runner(parameters=parameters)

    BackUp.save_data(results, parameters=parameters, graphics=0)
        

 
                        
if __name__ == "__main__":

    simple_main()


# Bonsoir 
