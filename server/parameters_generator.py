import numpy as np
import pickle
from datetime import datetime 


class ParametersGenerator(object):

    def __init__(self):

        self.alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.tau_list = [0.01, 0.02, 0.03, 0.04, 0.05]
        self.vision_list = [1, 3, 6, 12, 15]
        self.area_list = [1, 3, 6, 12, 15]

        self.stride = 1
        self.t_max = 600
        self.width = 30
        self.height = 30

        self.workforce_step = 5
        self.workforce_mini = 50
        self.workforce_maxi = 70

    def generate_workforce_list(self):

        array = np.zeros(3)
        array[:] = self.workforce_mini

        workforce_list = list()
        workforce_list.append(array.copy())

        while array[0] < self.workforce_maxi:

            for i in range(len(array)):
                array[i] += self.workforce_step
                workforce_list.append(array.copy())

        return workforce_list

    def generate_parameters_list(self, workforce_list):

        idx = 0
        parameters_list = []

        for workforce in workforce_list:

            for alpha in self.alpha_list:

                for tau in self.tau_list:

                    for vision in self.vision_list:

                        for area in self.area_list:
                            parameters = \
                                {
                                    "workforce": np.array(workforce, dtype=int),
                                    "alpha": alpha,  # Set the coefficient learning
                                    "tau": tau,  # Set the softmax parameter.
                                    "t_max": self.t_max,  # Set the number of time units the simulation will run
                                    # by each agent a at each round
                                    "stride": self.stride,
                                    "vision": vision,  # Set the importance of other agents'results in
                                    # front of an indivual res
                                    "area": area,  # set the perimeter within each agent can move
                                    "map_limits": {"width": self.width, "height": self.height},
                                    "idx": idx

                                }

                            idx += 1
                            parameters_list.append(parameters)

        return parameters_list

    @classmethod
    def generate_parameters_dict(cls, parameters_list):

        parameters_dict = {}

        nb_sub_list = 10
        sub_part = int(len(parameters_list) / 10)

        cursor = 0

        for i in range(nb_sub_list):

            part = parameters_list[cursor:cursor + sub_part]
            parameters_dict[i] = part
            cursor += sub_part

        while cursor < len(parameters_list):

            for i in range(nb_sub_list):

                if cursor < len(parameters_list):
                    parameters_dict[i].append(parameters_list[cursor])
                    cursor += 1

        return parameters_dict

    @classmethod
    def save_parameters_dict(cls, parameters_dict):

        date = str(datetime.now())[:-10].replace(" ", "_").replace(":", "-")

        for i in parameters_dict.keys():
            pickle.dump(parameters_dict[i],
                        open("../../data/parameters_lists/slices_{d}_{i}.p".format(i=i, d=date), mode="wb"))

    def run(self):

        workforce_list = self.generate_workforce_list()

        parameters_list = self.generate_parameters_list(workforce_list=workforce_list)

        parameters_dict = self.generate_parameters_dict(parameters_list)

        self.save_parameters_dict(parameters_dict)


def main():

    p = ParametersGenerator()
    p.run()
                        
if __name__ == "__main__":

    main()

