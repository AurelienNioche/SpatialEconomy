import numpy as np
import pickle
from datetime import datetime 
from os import path, mkdir
import re


class ParametersGenerator(object):

    def __init__(self):

        self.alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        self.tau_list = [0.01, 0, 0.15, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        self.vision_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        self.area_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

        self.stride = 1
        self.t_max = 1000
        self.width = 30
        self.height = 30

        self.workforce_step = 10
        self.workforce_mini = 50
        self.workforce_maxi = 150
    
        self.date = str(datetime.now())[:-10].replace(" ", "_").replace(":", "-")

        self.nb_sub_list = 500

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
        suffixes_list = [] 

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
                                    "vision": vision,
                                    "area": area,  # set the perimeter within each agent can move
                                    "map_limits": {"width": self.width, "height": self.height},
                                    "idx": idx,
                                    "date": self.date

                                }
                            parameters_list.append(parameters)
                            suffixes_list.append("{date}_idx{idx}".format(date=self.date, idx=idx))

                            # incremente idx
                            idx += 1

        return parameters_list, suffixes_list

    def generate_parameters_dict(self, parameters_list):

        parameters_dict = {}

        sub_part = int(len(parameters_list) / self.nb_sub_list)

        cursor = 0

        for i in range(self.nb_sub_list):

            part = parameters_list[cursor:cursor + sub_part]
            parameters_dict[i] = part
            cursor += sub_part

        while cursor < len(parameters_list):

            for i in range(self.nb_sub_list):

                if cursor < len(parameters_list):
                    parameters_dict[i].append(parameters_list[cursor])
                    cursor += 1

        return parameters_dict

    def save_parameters_dict(self, parameters_dict, suffixes_list):

        folders = ["../../data/parameters_lists/", "../../data/session/"]
        
        for i in folders:
            if not path.exists(i):
                mkdir(i)
        
        for i in parameters_dict.keys():
            pickle.dump(parameters_dict[i],
                        open("../../data/parameters_lists/slice_{}.p".format(i), mode="wb"))

        pickle.dump(suffixes_list,  open("../../data/session/session_{}.p".format(self.date), mode="wb"))

    def create_scripts(self):

        directory = "../../data/session/"
        root_file = "simulation.sh"
        prefix_output_file = "{}basile-simulation_".format(directory)

        if not path.exists(directory):
            mkdir(directory)

        for i in range(self.nb_sub_list):
            f = open(root_file, 'r')
            content = f.read()
            f.close()

            replaced = re.sub('slice_0', 'slice_{}'.format(i), content)
            replaced = re.sub('SimuBasile', 'SimuBasile{}'.format(i), replaced)

            f = open("{}{}.sh".format(prefix_output_file, i), 'w')
            f.write(replaced)
            f.close()

    def create_meta_launcher(self):

        content = "# !/usr/bin/env bash\n" \
                  "for i in {0..%d}; do\nqsub basile-simulation_${i}.sh \ndone" % (self.nb_sub_list - 1)

        directory = "../../data/session/"
        f = open("{}meta_launcher.sh".format(directory), 'w')
        f.write(content)
        f.close()
    
    def run(self):

        workforce_list = self.generate_workforce_list()

        parameters_list, suffixes_list = self.generate_parameters_list(workforce_list=workforce_list)

        parameters_dict = self.generate_parameters_dict(parameters_list)

        self.save_parameters_dict(parameters_dict, suffixes_list)

        self.create_scripts()

        self.create_meta_launcher()


def main():

    p = ParametersGenerator()
    p.run()
                        
if __name__ == "__main__":

    main()

