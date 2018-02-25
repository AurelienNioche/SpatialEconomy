import datetime
import pickle
import json
import os


class Parameters:

    def __init__(self, vision_area, movement_area, stride, x0, x1, x2,
                 alpha, tau, map_width, map_height, t_max, seed):
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
        self.file_name = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")

    def save(self):

        with open("{}{}.p".format(self.pickle_folder, self.file_name), 'wb') as f:
            pickle.dump(self, f)

        with open("{}{}.json".format(self.json_folder, self.file_name), 'w') as f:
            json.dump(self.parameters.__dict__, f, indent=2)
