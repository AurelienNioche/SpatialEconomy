import numpy as np
from itertools import product
import sys, time
import pickle
import matplotlib.pyplot as plt
from module.converter import write
from tqdm import tqdm
from multiprocessing import Pool
import datetime


############################################
#           NOTATION                       #
############################################

# For the needs of coding, we don't use systematically here the same notation as in the article.
# Here are the matches:

# For an object:
# 'i' means a production good;
# 'j' means a consumption good;
# 'k' means the third good.

# For agent type:
# * '0' means a type-12 agent;
# * '1' means a type-22 agent;
# * '2' means a type-31 agent.

# For a decision:
# * '0' means 'type-i decision';
# * '1' means 'type-k decision'.

# For a choice:
# * '0' means 'ij' if the agent faces a type-i decision and 'kj' if the agent faces a type-k decision;
# * '1' means 'ik'  if the agent faces a type-i decision and 'ki'  if the agent faces type-k decision.

# For markets,
# * '0' means the part of the market '12' where are the agents willing
#       to exchange type-1 good against type-2 good;
# * '1' means the part of the market '12' where are the agents willing
#       to exchange type-2 good against type-1 good;
# * '2' means the part of the market '23' where are the agents willing
#       to exchange type-2 good against type-3 good;
# * '3' means the part of the market '23' where are the agents willing
#       to exchange type-3 good against type-2 good;
# * '4' means the part of the market '31' where are the agents willing
#       to exchange type-3 good against type-1 good;
# * '5' means the part of the market '31' where are the agents willing
#       to exchange type-1 good against type-3 good.


class Economy(object):
    def __init__(self, parameters):

        self.vision = parameters["vision"]

        self.area = parameters["area"]
        self.stride = parameters["stride"]

        self.map_limits = parameters["map_limits"]

        self.absolute_matrix = np.array([[[0, 1], [1, 2], [2, 0]],
                                         [[0, 2], [1, 0], [2, 1]],
                                         [[2, 1], [0, 2], [1, 0]],
                                         [[2, 0], [0, 1], [1, 2]]], dtype=object)

        self.absolute_exchange_to_int = \
            {
                (0, 1): 0,
                (0, 2): 1,
                (1, 0): 2,
                (1, 2): 3,
                (2, 0): 4,
                (2, 1): 5,

            }

        # # i: type; j: i_choice
        self.int_to_relative_choice = np.array([
            [0, 3, -1],
            [1, 2, -1],
            [-1, 0, 3],
            [3, -1, 0],
            [2, -1, 1]], dtype=int)


        # To convert relative choices into absolute choices
        # 0 : 0 -> 1
        # 1 : 0 -> 2
        # 2 : 1 -> 0
        # 3 : 1 -> 2
        # 4 : 2 -> 0
        # 5 : 2 -> 1

        # i: type; j: i_choice
        self.relative_to_absolute_choice = np.array([
            [0, 1, 5, 4],
            [3, 2, 1, 0],
            [4, 5, 2, 3]], dtype=int)

        self.n = np.sum(parameters["workforce"])  # Total number of agents
        self.workforce = np.zeros(len(parameters["workforce"]), dtype=int)
        self.workforce[:] = parameters["workforce"]  # Number of agents by type

        self.alpha = parameters["alpha"]  # Learning coefficient
        self.temperature = parameters["tau"]  # Softmax parameter

        self.type = np.zeros(self.n, dtype=int)
        self.good = np.zeros(self.n, dtype=int)

        self.type[:] = np.concatenate(([0, ] * self.workforce[0],
                                       [1, ] * self.workforce[1],
                                       [2, ] * self.workforce[2]))
        self.good[:] = np.concatenate(([0, ] * self.workforce[0],
                                       [1, ] * self.workforce[1],
                                       [2, ] * self.workforce[2]))
        self.map_of_agents = dict()

        self.saving_map = dict()
        # Each agent possesses an index by which he can be identified.
        #  Here are the the indexes lists corresponding to each type of agent:

        self.idx0 = np.where(self.type == 0)[0]
        self.idx1 = np.where(self.type == 1)[0]
        self.idx2 = np.where(self.type == 2)[0]

        self.position = [(-1, -1) for i in range(self.n)]

        self.x_perimeter = np.zeros(self.n, dtype=[("min", int, 1),
                                                   ("max", int, 1)])
        self.y_perimeter = np.zeros(self.n, dtype=[("min", int, 1),
                                                   ("max", int, 1)])

        self.decision = np.zeros(self.n, dtype=int)

        self.choice = np.zeros(self.n, dtype=int)
        self.i_choice = np.zeros(self.n, dtype=int)

        # Values for each option of choice.
        # The 'option0' and 'option1' are just the options that are reachable by the agents at time t,
        #  among the four other options.
        self.value_ij = np.zeros(self.n)
        self.value_ik = np.zeros(self.n)
        self.value_kj = np.zeros(self.n)
        self.value_ki = np.zeros(self.n)
        self.value_option0 = np.zeros(self.n)
        self.value_option1 = np.zeros(self.n)

        # Initialize the estimations of easiness of each agents and for each type of exchange.
        self.estimation = np.zeros((4, self.n))

        self.exchange_matrix = \
            np.zeros((self.map_limits["width"], self.map_limits["height"]),
                     dtype=[("0", float, 1), ("1", float, 1), ("2", float, 1)])

        for i in [0, 1, 2]:
            self.exchange_matrix[str(i)] = np.zeros((self.map_limits["width"], self.map_limits["height"]))

        self.choices_list = {"0": list(), "1": list(), "2": list()}

        self.t = 0

        self.direct_choices_proportions = {"0": 0., "1": 0., "2": 0.}
        self.direct_exchange = {"0": 0., "1": 0., "2": 0.}
        self.indirect_exchange = {"0": 0., "1": 0., "2": 0.}

        # This is the initial guest (same for every agent).
        # '1' means each type of exchange can be expected to be realized in only one unit of time
        # The more the value is close to zero, the more an exchange is expected to be hard.
        #
        self.estimation[:] = np.random.random((4, self.n))

    # --------------------------------------------------||| SETUP |||----------------------------------------------- #

    def setup(self):

        self.setup_insert_agents_on_map()
        self.setup_define_perimeters()

    def setup_insert_agents_on_map(self):

        for idx in range(self.n):

            while True:
                x = np.random.randint(0, self.map_limits["width"])
                y = np.random.randint(0, self.map_limits["height"])

                if (x, y) not in self.position:
                    self.position[idx] = (x, y)
                    self.map_of_agents[(x, y)] = idx
                    self.saving_map[(x, y)] = (self.type[idx], self.good[idx])

                    break

    def setup_define_perimeters(self):

        for idx in range(self.n):
            self.x_perimeter["min"][idx] = self.position[idx][0] - self.area
            self.x_perimeter["max"][idx] = self.position[idx][0] + self.area
            self.y_perimeter["min"][idx] = self.position[idx][1] - self.area
            self.y_perimeter["max"][idx] = self.position[idx][1] + self.area

    # --------------------------------------------------||| RESET |||----------------------------------------------- #

    def reset(self):

        for i in self.direct_exchange.keys():
            self.direct_exchange[i] = 0.
            self.indirect_exchange[i] = 0.


        # ---------------------------------------------||| MOVE /  MAP OPERATIONS |||------------------------------------ #

    def move(self, idx):

        positions_in_map = self.move_check_nearby_positions(idx)  # Main method
        self.move_find_free_position(idx, positions_in_map)

    def move_check_nearby_positions(self, idx):

        # Method used in order to find 1)free positions around current
        #  agent
        # 2)occupied positions around
        position = self.position[idx]

        nearby_positions = np.asarray([(position[0] + i[0],
                                        position[1] + i[1]) for i in product([-1, 0, 1], repeat=2)])

        # nearby_positions is a matrix of 9*2 (9: number of cases around the agent, included his own position,
        # 2: x, y coordinates)

        # We look at x  and y columns to check that they are in map dimensions
        result_x_inf = nearby_positions[:, 0] < self.map_limits["width"]
        result_y_inf = nearby_positions[:, 1] < self.map_limits["height"]

        result_x_sup = nearby_positions[:, 0] >= 0
        result_y_sup = nearby_positions[:, 1] >= 0

        b_positions_in_map_x = result_x_inf * result_x_sup
        b_positions_in_map_y = result_y_inf * result_y_sup

        b_positions_in_map = b_positions_in_map_x * b_positions_in_map_y

        positions_in_map = nearby_positions[b_positions_in_map]

        # test if they are in the perimeter of the agent

        result_x_inf = positions_in_map[:, 0] < self.x_perimeter["max"][idx]
        result_x_sup = positions_in_map[:, 0] >= self.x_perimeter["min"][idx]

        result_y_inf = positions_in_map[:, 1] < self.y_perimeter["max"][idx]
        result_y_sup = positions_in_map[:, 1] >= self.y_perimeter["min"][idx]

        b_positions_in_map_x = result_x_inf * result_x_sup
        b_positions_in_map_y = result_y_inf * result_y_sup

        b_positions_in_map = b_positions_in_map_x * b_positions_in_map_y

        return positions_in_map[b_positions_in_map]

    def move_find_free_position(self, idx, positions_in_map):

        np.random.shuffle(positions_in_map)

        for i in positions_in_map:

            i = tuple(i)

            # If position
            if i not in self.position:
                # Update dics
                self.map_of_agents[i] = self.map_of_agents.pop(tuple(self.position[idx]))
                self.saving_map[i] = self.saving_map.pop(tuple(self.position[idx]))

                # Agent takes i as new position
                self.position[idx] = i

                break

    # ------------------------------------------------||| MAKE ENCOUNTER |||--------------------------------------- #

    def encounter(self, idx):

        occupied_nearby_positions = self.encounter_check_nearby_positions(idx)
        group_idx = self.encounter_look_for_partners(idx, occupied_nearby_positions)
        choice_current_agent, proportion_of_matching_choices, partner_id = \
            self.encounter_look_for_partners_choices(idx, group_idx)

        self.encounter_update_estimations(idx=idx, group_idx=group_idx,
                                          acceptance_frequency=proportion_of_matching_choices,
                                          exchange_type=choice_current_agent)
        if partner_id is not None:

            self.encounter_proceed_to_exchange(idx, partner_id)

    def encounter_check_nearby_positions(self, idx):

        nearby_positions_x = {"min": self.position[idx][0] - self.vision,
                              "max": self.position[idx][0] + self.vision}
        nearby_positions_y = {"min": self.position[idx][1] - self.vision,
                              "max": self.position[idx][1] + self.vision}

        position = np.asarray(self.position)

        result_x_inf = position[:, 0] <= nearby_positions_x["max"]
        result_y_inf = position[:, 1] <= nearby_positions_y["max"]
        result_x_sup = position[:, 0] >= nearby_positions_x["min"]
        result_y_sup = position[:, 1] >= nearby_positions_y["min"]

        b_occupied_nearby_positions_x = result_x_inf * result_x_sup
        b_occupied_nearby_positions_y = result_y_inf * result_y_sup

        b_occupied_nearby_positions = b_occupied_nearby_positions_x * b_occupied_nearby_positions_y

        occupied_nearby_positions = position[b_occupied_nearby_positions]

        return occupied_nearby_positions

    def encounter_look_for_partners(self, idx, positions_in_map):

        idx_informers = []
        for i in positions_in_map:
            i = tuple(i)
            idx_informers[idx].append(self.map_of_agents[i])
        return idx_informers

    def encounter_look_for_partners_choices(self, idx, group_idx):
       
        # The agent chooses the good he wants to obtain and asks agents around him for it  
        
        self.choose(idx)
        choice_current_agent = list(self.absolute_matrix[self.i_choice[idx], self.type[idx]])
        int_choice_current_agent = self.absolute_exchange_to_int[tuple(choice_current_agent)]

        matching_list = list()
        
        # We retrieve the good wanted by the others and check if their needs and our agent need match

        partner_ids = []
        
        for partner_id in group_idx:

            self.choose(partner_id)
            choice_current_partner = list(self.absolute_matrix[self.i_choice[partner_id], self.type[partner_id]])
            success = choice_current_partner[::-1] == choice_current_agent
            matching_list.append(success)
            if success:
                partner_ids.append(partner_id)

        if partner_ids:

            partner_id = np.random.choice(partner_ids)
        else:
            partner_id = None

        proportion_of_matching_choices = np.mean(matching_list)
        
        return int_choice_current_agent, proportion_of_matching_choices, partner_id

    def encounter_proceed_to_exchange(self, idx, partner_id):

        self.good[idx], self.good[partner_id] = self.good[partner_id], self.good[idx]

        # If they succeeded getting  their consumption good, they consume it directly.

        if self.i_choice[idx] in [0, 3]:
            self.good[idx] = self.type[idx]

        if self.i_choice[partner_id] in [0, 3]:
            self.good[partner_id] = self.type[partner_id]

        for i in [idx, partner_id]:

            self.decision[i] = self.i_choice[i] == 1

        # ----------- #
        # Saving....
        # ----------- #

        self.saving_map[tuple(self.position[idx])] = (self.type[idx], self.good[idx])
        self.saving_map[tuple(self.position[partner_id])] = (self.type[partner_id], self.good[partner_id])

    def encounter_update_estimations(self, idx, group_idx, acceptance_frequency, exchange_type):

        for idx in group_idx:

            relative_choice = self.int_to_relative_choice[self.type[idx], exchange_type]

            self.estimation[relative_choice, idx] += \
                self.alpha(acceptance_frequency - self.estimation[relative_choice, idx])

    def encounter_exchange_count(self, idx, partner_id):

        exchange_position = self.position[partner_id]

        if self.good[idx] + self.good[partner_id] == 1:

            self.exchange_matrix["0"][exchange_position] += 1

        elif self.good[idx] + self.good[partner_id] == 3:

            self.exchange_matrix["1"][exchange_position] += 1

        else:

            self.exchange_matrix["2"][exchange_position] += 1

    # ----------------------------------------------------||| CHOICE ||| -------------------------------------------- #

    def choose(self, idx):

        self.choose_update_options_values(idx)
        self.choose_decision_rule(idx)

    def choose_update_options_values(self, idx):

        # Each agent try to minimize the time to consume
        # That is v(option) = 1/(1/estimation)

        # Set value to each option choice

        self.value_ij[idx] = self.estimation[0, idx]
        self.value_kj[idx] = self.estimation[2, idx]

        if not (self.estimation[1, idx] + self.estimation[2, idx]) == 0:

            self.value_ik[idx] = \
                (self.estimation[1, idx] * self.estimation[2, idx]) / \
                (self.estimation[1, idx] + self.estimation[2, idx])
        else:  # Avoid division by 0
            self.value_ik[idx] = 0

        if not (self.estimation[3, idx] + self.estimation[0, idx]) == 0:
            self.value_ki[idx] = \
                (self.estimation[3, idx] * self.estimation[0, idx]) / \
                (self.estimation[3, idx] + self.estimation[0, idx])
        else:  # Avoid division by 0
            self.value_ki[idx] = 0

    def choose_decision_rule(self, idx):

        if self.decision[idx] == 0:
            self.value_option0[idx] = self.value_ij[idx]
            self.value_option1[idx] = self.value_ik[idx]
        else:
            self.value_option0[idx] = self.value_kj[idx]
            self.value_option1[idx] = self.value_ki[idx]

        # Set a probability to current option 0 using softmax rule
        # (As there is only 2 options each time, computing probability for a unique option is sufficient)

        probability_of_choosing_option0 = \
            np.exp(self.value_option0[idx] / self.temperature) / \
            (np.exp(self.value_option0[idx] / self.temperature) +
             np.exp(self.value_option1[idx] / self.temperature))

        random_number = np.random.random()  # Generate random number

        # Make a choice using the probability of choosing option 0 and a random number for each agent
        # Choose option 1 if random number > or = to probability of choosing option 0,
        #  choose option 0 otherwise
        self.choice[idx] = random_number >= probability_of_choosing_option0[idx]
        self.i_choice[idx] = (self.decision[idx] * 2) + self.choice[idx]

        if self.i_choice[idx] == 0:

            self.direct_exchange[str(self.type[idx])] += 1

        else:

            self.indirect_exchange[str(self.type[idx])] += 1


    # ------------------------------------------------||| COMPUTE CHOICES PROPORTIONS |||---------------------------- #

    def compute_choices_proportions(self):

        '''
        Find proproption of choice leading to a direct exchange
        :return:
        '''

        for i in [0, 1, 2]:
            self.direct_choices_proportions[str(i)] = \
                self.direct_exchange[str(i)] / (self.direct_exchange[str(i)] +
                                                self.indirect_exchange[str(i)])

    def append_choices_to_compute_means(self):

        for i in [0, 1, 2]:
            self.choices_list[str(i)].append(self.direct_choices_proportions[str(i)])

    def compute_choices_means(self):

        list_mean = []

        for i in range(3):
            list_mean.append(np.mean(self.choices_list[str(i)]))

        return list_mean


# ---------------------------------------------||| MAIN RUNNER |||---------------------------------------------------- #
class SimulationRunner(object):
    @classmethod
    def main_runner(cls, parameters):

        # Create the economy to simulate

        result = cls.launch_economy(parameters)
        return result

    @staticmethod
    def launch_economy(parameters):

        print("Producing data...")

        eco = Economy(parameters)
        map_limits = parameters["map_limits"]

        list_saving_map = list()
        matrix_list = [[], ] * 3
        exchanges_proportions_list = list()

        # Place agents and stuff...
        eco.setup()

        # Save initial positions
        list_saving_map.append(eco.saving_map.copy())

        for t in tqdm(range(parameters["t_max"])):

            eco.reset()

            for idx in range(eco.n):

                # move agent, then make them proceeding to exchange
                eco.move(idx)
                eco.encounter(idx)

                # -------------------- #
                # For saving...
                # ------------------- #

                for i in range(3):
                    # create empty matrix
                    matrix = np.zeros((map_limits["height"], map_limits["width"]))

                    # fill the matrix with the exchanges positions
                    matrix[:] = eco.exchange_matrix[str(i)][:]

                    # For each "t" and each trial the matrix are added to a list
                    matrix_list[i].append(matrix.copy())

                # Same for graphics positions
                list_saving_map.append(eco.saving_map.copy())

                # -----------------  #

            # ---------- #
            # Do some stats...

            # for each "t" we compute the proportion of direct choices
            eco.compute_choices_proportions()

            # We append it to a list (in the fonction)
            eco.append_choices_to_compute_means()

            # We copy proportions and add them to a list
            proportions = eco.direct_choices_proportions.copy()
            exchanges_proportions_list.append(proportions)

        # Finally we compute the direct choices mean for each type
        # of agent and return it as well as the direct choices proportions

        list_mean = eco.compute_choices_means()

        result = {"exchanges_proportions_list": exchanges_proportions_list,
                  "list_mean": list_mean,
                  "matrix_list": matrix_list,
                  "list_map": list_saving_map}

        return result


class BackUp(object):
    @classmethod
    def save_data(cls, results, parameters, graphics=1):
        print("\nSaving data...")

        date = str(datetime.now())[:-10].replace(" ", "_").replace(":", "-")

        file = open("../data/last.txt", mode="w")
        file.write(date)
        file.close()

        # Save matrix of exchanges and positions dict (in order to print the main map later)

        if graphics:

            matrix_list = results["matrix_list"]
            list_map = results["list_map"]

            for i in range(len(matrix_list)):
                write(matrix_list[i], table_name="exchange_{i}".format(i=i),
                      database_name='array_exchanges{}'.format(date), descr="{}/3".format(i + 1))

            pickle.dump(list_map, open("../data/position_map{}.p".format(date), mode='wb'))

        exchanges_proportions_list = results["exchanges_proportions_list"]
        pickle.dump(exchanges_proportions_list, open("../data/exchanges{}.p".format(date), mode='wb'))
        pickle.dump(parameters, open("../data/parameters{}.p".format(date), mode='wb'))


def simple_main():
    '''
    Simplest program use
    :return: None
    '''

    parameters = \
        {
            "workforce": np.array([10, 10, 10], dtype=int),
            "alpha": 0.5,  # Set the coefficient learning
            "tau": 0.05,  # Set the softmax parameter.
            "t_max": 10,  # Set the number of time units the simulation will run
            "stride": 1,  # by each agent at each round
            "vision": 5,  # Set the importance of other agents'results in
            "area": 10,  # front of an individual res
            "map_limits": {"width": 20, "height": 20},

        }

    results = SimulationRunner.main_runner(parameters=parameters)

    BackUp.save_data(results, parameters=parameters, graphics=1)


if __name__ == "__main__":
    simple_main()
