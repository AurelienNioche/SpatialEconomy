import datetime
import pickle
import json
import os


class Parameters:

    def __init__(self, vision_area, movement_area, stride, x0, x1, x2,
                 alpha, tau, map_width, map_height, t_max, seed, graphics):
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.stride = stride
        self.movement_area = movement_area
        self.vision_area = vision_area
        self.alpha = alpha
        self.tau = tau
        self.map_width = map_width
        self.map_height = map_height
        self.t_max = t_max
        self.seed = seed
        self.graphics = graphics


class ParametersPool:
    
    def __init__(self, t_max, map_height, map_width,
                 alpha_min, alpha_max, tau_min, tau_max,
                 movement_area_min, movement_area_max,
                 vision_area_min, vision_area_max, x_min, x_max,
                 stride_min, stride_max, n, seed, graphics):
        
        self.t_max = t_max
        self.map_height = map_height
        self.map_width = map_width
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.movement_area_min = movement_area_min
        self.movement_area_max = movement_area_max
        self.vision_area_min = vision_area_min
        self.vision_area_max = vision_area_max
        self.x_min = x_min
        self.x_max = x_max
        self.stride_min = stride_min
        self.stride_max = stride_max
        self.n = n
        self.seed = seed
        self.graphics = graphics


class Result:

    data_folder = "data/"
    pickle_folder = data_folder + "pickle/"
    json_folder = data_folder + "json/"
    for f in (pickle_folder, json_folder):
        os.makedirs(f, exist_ok=True)

    def __init__(self, direct_exchanges_proportions, indirect_exchanges_proportions,
                 exchange_maps, agent_maps, parameters):

        self.direct_exchanges_proportions = direct_exchanges_proportions
        self.indirect_exchanges_proportions = indirect_exchanges_proportions
        self.exchange_maps = exchange_maps
        self.agent_maps = agent_maps
        self.parameters = parameters
        self.file_name = datetime.datetime.now().strftime("single_%y_%m_%d_%H_%M_%S_%f")

    def save(self):

        with open("{}{}.p".format(self.pickle_folder, self.file_name), 'wb') as f:
            pickle.dump(self, f)

        with open("{}{}.json".format(self.json_folder, self.file_name), 'w') as f:
            json.dump(self.parameters.__dict__, f, indent=2)


class ResultPool:

    data_folder = "data/"
    pickle_folder = data_folder + "pickle/"
    json_folder = data_folder + "json/"
    for f in (pickle_folder, json_folder):
        os.makedirs(f, exist_ok=True)

    def __init__(self, data, parameters):

        self.data = data
        self.parameters = parameters

        self.file_name = datetime.datetime.now().strftime("pool_%y_%m_%d_%H_%M_%S_%f")

    def save(self):

        file_name = "{}{}.p".format(self.pickle_folder, self.file_name)
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(self, f)
        except OSError:
            print("Could not save '{}'".format(file_name))

        with open("{}{}.json".format(self.json_folder, self.file_name), 'w') as f:
            json.dump(self.parameters.__dict__, f, indent=2)
