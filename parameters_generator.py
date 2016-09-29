import numpy as np
import pickle
from datetime import datetime 
from os import path, mkdir
import re
import shutil
from arborescence.arborescence import Folders


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

        self.nb_sub_list = 1000

        self.folders = Folders.folders

        self.root_file = "simulation_template.sh"

    def empty_scripts_folder(self):

        if path.exists(self.folders["data"]):

            response = input("Do you want to remove data folder?")

            while response not in ['y', 'yes', 'n', 'no', 'Y', 'N']:
                response = input("You can only respond by 'yes' or 'no'.")

            print("Proceeding...")

            if response in ['y', 'yes', 'Y']:

                if path.exists(self.folders["data"]):
                    shutil.rmtree(self.folders["data"])
                print("Data folder has been erased.")
            else:
                print("Data folder has been conserved.")

        else:
            print("Proceeding...")

        print("Remove old scripts and logs...")

        if path.exists(self.folders["scripts"]):
            shutil.rmtree(self.folders["scripts"])

        print("Old scripts and logs have been removed.")

        if path.exists(self.folders["logs"]):
            shutil.rmtree(self.folders["logs"])

    def create_folders(self):

        for directory in self.folders.values():

            if not path.exists(directory):
                mkdir(directory)

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
                                    "vision": vision,
                                    "area": area,  # set the perimeter within each agent can move
                                    "map_limits": {"width": self.width, "height": self.height},
                                    "idx": idx,
                                    "date": self.date

                                }
                            parameters_list.append(parameters)
                            idx += 1

        return parameters_list

    def generate_parameters_dict(self, parameters_list):

        parameters_dict = {}

        sub_part = int(len(parameters_list) / self.nb_sub_list)
        print("N simulations:", len(parameters_list))
        print("N slices of input parameters:", self.nb_sub_list)
        print("N simulations parameters per slice:", sub_part)

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

    def save_parameters_dict(self, parameters_dict):
        
        for i in parameters_dict.keys():
            pickle.dump(parameters_dict[i],
                        open("{}/slice_{}.p".format(self.folders["parameters"], i), mode="wb"))

    def create_scripts(self):

        prefix_output_file = "{}/spatial-simulation_".format(self.folders["scripts"])

        for i in range(self.nb_sub_list):
            f = open(self.root_file, 'r')
            content = f.read()
            f.close()

            replaced = re.sub('slice', 'slice_{}'.format(i), content)
            replaced = re.sub('SimuSpatial', 'SimuSpatial{}'.format(i), replaced)

            f = open("{}{}.sh".format(prefix_output_file, i), 'w')
            f.write(replaced)
            f.close()

    # def create_meta_launcher(self):
    #
    #     content = "# !/usr/bin/env bash\n" \
    #               "for i in {0..%d}; do\nqsub spatial-simulation_${i}.sh \ndone" % (self.nb_sub_list - 1)
    #
    #     f = open("{}/meta_launcher.sh".format(self.folders["scripts"]), 'w')
    #     f.write(content)
    #     f.close()
    #
    def run(self):

        self.empty_scripts_folder()
        self.create_folders()

        print("Generate parameters list...")

        workforce_list = self.generate_workforce_list()

        parameters_list = self.generate_parameters_list(workforce_list=workforce_list)

        parameters_dict = self.generate_parameters_dict(parameters_list)

        self.save_parameters_dict(parameters_dict)
        print("Generate scripts...")
        self.create_scripts()
        # print("Generate launcher...")
        #
        # self.create_meta_launcher()
        print("Done.")


def main():

    p = ParametersGenerator()
    p.run()
                        
if __name__ == "__main__":

    main()

