from eco.c_economy import SimulationRunner
from save.save_eco import BackUp


def main():

    parameters = {
        "vision_area": 30,
        "movement_area": 30,
        "stride": 1,
        "width": 30,
        "height": 30,
        "x0": 150,
        "x1": 150,
        "x2": 150,
        "alpha": 0.1,
        "tau": 0.05,
        "t_max": 2
    }
    results = SimulationRunner.launch_economy(parameters=parameters, graphics=None)
    BackUp.save_data(results=results, parameters=parameters, graphics=None)


if __name__ == "__main__":

    main()
