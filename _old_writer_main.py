import numpy as np
from itertools import product
import sys, time
import pickle
import matplotlib.pyplot as plt
import converter
from tqdm import tqdm
from multiprocessing import Pool


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

        self.choice_transition = np.array([
            [0, 1, 2, 3],
            [-1, -1, 1, 0],
            [3, 2, -1, -1]])

        self.n = np.sum(parameters["workforce"])  # Total number of agents
        self.workforce = np.zeros(len(parameters["workforce"]), dtype=int)
        self.workforce[:] = parameters["workforce"]  # Number of agents by type

        self.alpha = parameters["alpha"]  # Learning coefficient
        self.temperature = parameters["tau"]  # Softmax parameter

        self.money_threshold = parameters["money_threshold"]
        self.money_delta = parameters["money_delta"]
        self.welfare_delta = parameters["welfare_delta"]

        self.total_received_information = np.zeros(self.n, dtype=int)
        self.epsilon = parameters["epsilon"]

        self.type = np.zeros(self.n, dtype=int)
        self.good = np.zeros(self.n, dtype=int)

        self.type[:] = np.concatenate(([0, ] * self.workforce[0],
                                       [1, ] * self.workforce[1],
                                       [2, ] * self.workforce[2]))
        self.good[:] = np.concatenate(([0, ] * self.workforce[0],
                                       [1, ] * self.workforce[1],
                                       [2, ] * self.workforce[2]))
        self.map_of_agents = dict()

        self.graphic_map = dict()
        # Each agent possesses an index by which he can be identified.
        #  Here are the the indexes lists corresponding to each type of agent:

        self.idx0 = np.where(self.type == 0)[0]
        self.idx1 = np.where(self.type == 1)[0]
        self.idx2 = np.where(self.type == 2)[0]

        # self.position = np.zeros((self.n, 2), dtype=[("x", int, 1), ("y", int, 1)])

        self.position = [(i, i) for i in range(self.n)]

        self.x_perimeter = np.zeros((self.n, self.area * 2))
        self.y_perimeter = np.zeros((self.n, self.area * 2))

        # The "decision array" is a 3D-matrix (d1: finding_a_partner, d2: decision, d3: choice).
        # Allow us to retrieve the decision faced by an agent at t according to
        #  * the fact that he succeeded in his exchange at t-1,
        #  * the decision he faced at t-1,
        #  * the choice he made at t-1.
        self.decision_array = np.array(
            [[[0, 0],
              [1, 1]],
             [[0, 1],
              [0, 0]]])
        self.absolute_matrix = np.array([[[0, 1], [1, 2], [2, 0]],
                                         [[0, 2], [1, 0], [2, 1]],
                                         [[2, 1], [0, 2], [1, 0]],
                                         [[2, 0], [0, 1], [1, 2]]], dtype=object)

        self.decision = np.zeros(self.n, dtype=int)

        self.choice = np.zeros(self.n, dtype=int)

        self.random_number = np.zeros(self.n, dtype=float)  # Used for taking a decision

        self.probability_of_choosing_option0 = np.zeros(self.n, dtype=float)

        self.finding_a_partner = np.zeros(self.n, dtype=int)

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
        self.estimation_ik = np.zeros(self.n)
        self.estimation_ij = np.zeros(self.n)
        self.estimation_kj = np.zeros(self.n)
        self.estimation_ki = np.zeros(self.n)

        self.exchange_matrix = np.zeros((self.map_limits["width"], self.map_limits["height"]), \
                                        dtype=[("0", float, 1), ("1", float, 1), ("2", float, 1)])

        for i in [0, 1, 2]:
            self.exchange_matrix[str(i)] = np.zeros((self.map_limits["width"], self.map_limits["height"]))

        self.choices_list = {"0": list(), "1": list(), "2": list()}

        # This is the initial guest (same for every agent).
        # '1' means each type of exchange can be expected to be realized in only one unit of time
        # The more the value is close to zero, the more an exchange is expected to be hard.
        self.t = 0

        self.t_emergence = -1

        self.direct_exchange = {"0": 0., "1": 0., "2": 0.}
        self.indirect_exchange = {"0": 0., "1": 0., "2": 0.}

        self.money = -1

        # Number of rounds, a good serve as money
        self.money_round = np.zeros(len(self.workforce))
        self.money_time = np.zeros(len(self.workforce))

        self.t_beginning_reward = 0
        self.t_end_reward = 0
        self.beginning_reward = np.zeros((self.welfare_delta, self.n), dtype=int)
        self.end_reward = np.zeros((self.welfare_delta, self.n), dtype=int)
        self.beginning_welfare = np.zeros(len(self.workforce))
        self.end_welfare = np.zeros(len(self.workforce))

        self.insert_agents_on_map()
        self.define_perimeters()
        self.estimation_ij[:] = np.random.random(self.n)
        self.estimation_ik[:] = np.random.random(self.n)
        self.estimation_kj[:] = np.random.random(self.n)
        self.estimation_ki[:] = np.random.random(self.n)

    # --------------------------------------------------||| CHECK NEARBY POSITIONS |||--------------------------------------------------------------------------#

    def check_nearby_positions(self, id, move_or_look_around):

        if move_or_look_around == "move":
            # Method used in order to find 1)free positions around current
            #  agent
            # 2)occupied positions around
            position = self.position[id]

            nearby_positions = np.asarray([(position[0] + i[0],
                                            position[1] + i[1]) for i in product([-1, 0, 1], repeat=2)])

            result_x = nearby_positions[:, 0] < self.map_limits["width"]
            result_y = nearby_positions[:, 1] < self.map_limits["height"]
            result_x_2 = nearby_positions[:, 0] >= 0
            result_y_2 = nearby_positions[:, 1] >= 0
            b_positions_in_map_x = result_x * result_x_2
            b_positions_in_map_y = result_y * result_y_2
            b_positions_in_map = b_positions_in_map_x * b_positions_in_map_y
            positions_in_map = nearby_positions[b_positions_in_map]



        else:

            nearby_positions_x = np.array([self.position[id][0] - self.vision, self.position[id][0] + self.vision])
            nearby_positions_y = np.array([self.position[id][1] - self.vision, self.position[id][1] + self.vision])

            position = np.asarray(self.position)

            result_x = position[:, 0] <= np.amax(nearby_positions_x)
            result_y = position[:, 1] <= np.amax(nearby_positions_y)
            result_x_2 = position[:, 0] >= np.amin(nearby_positions_x)
            result_y_2 = position[:, 1] >= np.amin(nearby_positions_x)
            b_positions_in_map_x = result_x * result_x_2
            b_positions_in_map_y = result_y * result_y_2
            b_positions_in_map = b_positions_in_map_x * b_positions_in_map_y
            positions_in_map = position[b_positions_in_map]

        return positions_in_map

    # --------------------------------------------------||| MOVE /  MAP OPERATIONS |||-----------------------------------------------------------------------------------------------------#

    def insert_agents_on_map(self):

        for id in range(self.n):

            while True:
                x = np.random.randint(0, self.map_limits["width"])
                y = np.random.randint(0, self.map_limits["height"])

                if (x, y) not in self.position:
                    self.position[id] = (x, y)
                    self.map_of_agents[(x, y)] = id
                    self.graphic_map[(x, y)] = (self.type[id], self.good[id])

                    break

    def define_perimeters(self):

        for id in range(self.n):
            self.x_perimeter[id] = np.arange(self.position[id][0] - self.area, self.position[id][0] + self.area)
            self.y_perimeter[id] = np.arange(self.position[id][1] - self.area, self.position[id][1] + self.area)

    def move_find_free_position(self, id, positions_in_map):

        np.random.shuffle(positions_in_map)

        for i in positions_in_map:

            i = tuple(i)

            if i not in self.position and i[0] in self.x_perimeter[id] and i[1] in self.y_perimeter[id]:
                self.map_of_agents[i] = self.map_of_agents.pop(tuple(self.position[id]))
                self.graphic_map[i] = self.graphic_map.pop(tuple(self.position[id]))
                self.position[id] = i

    def move(self, id):

        positions_in_map = self.check_nearby_positions(id, "move")  # Main method
        self.move_find_free_position(id, positions_in_map)

    # ------------------------------------------------||| MAKE ENCOUNTER |||---------------------------------------------------------------------------------------------------------#

    def encounter_update(self, id, partner_id):

        self.good[id], self.good[partner_id] = self.good[partner_id], self.good[id]

        self.graphic_map[tuple(self.position[id])] = (self.type[id], self.good[id])
        self.graphic_map[tuple(self.position[partner_id])] = (self.type[partner_id], self.good[partner_id])

        self.finding_a_partner[id], self.finding_a_partner[partner_id] = 1, 1

        # If they succeeded getting  their consumption good, they consume it directly.

        if self.i_choice[id] == 0 or self.i_choice[id] == 3:
            self.good[id] = self.type[id]

        if self.i_choice[partner_id] == 0 or self.i_choice[partner_id] == 3:
            self.good[partner_id] = self.type[partner_id]

    def encounter_look_for_partners(self, positions_in_map):

        self.information = list()

        for i in positions_in_map:
            i = tuple(i)
            self.information.append(self.map_of_agents[i])

    def encounter_exchange_count(self, id, partner_id):

        exchange_position = self.position[partner_id]

        if self.good[id] + self.good[partner_id] == 1:

            self.exchange_matrix["0"][exchange_position] += 1.

        elif self.good[id] + self.good[partner_id] == 3:

            self.exchange_matrix["1"][exchange_position] += 1.

        else:

            self.exchange_matrix["2"][exchange_position] += 1.

    def encounter_ask_for_exchange(self, id):  # Some restructuration could help

        self.make_a_choice(id)
        choice_current_agent = self.absolute_matrix[self.i_choice[id], self.type[id]]

        for partner_id in self.information:

            self.make_a_choice(partner_id)

            choice_current_partner = self.absolute_matrix[self.i_choice[partner_id], self.type[partner_id]]

            self.finding_a_partner[id], self.finding_a_partner[partner_id] = 0, 0

            if np.all(choice_current_partner[::-1] == choice_current_agent):
                return partner_id

    def encounter(self, id):

        positions_in_map = self.check_nearby_positions(id, "look_around")
        self.encounter_look_for_partners(positions_in_map)
        partner_id = self.encounter_ask_for_exchange(id)  # Main method

        if partner_id is not None:
            self.encounter_update(id, partner_id)
            self.update_estimations(partner_id)
            self.update_estimations(id)
            self.encounter_exchange_count(id, partner_id)


            # ----------------------------------------------------||| COMPUTE CHOICE, ESTIMATIONS ||| -----------------------------------------------------------------------------------------------------#

    def initialize_estimations(self):

        estimations = np.ones(self.n)
        for i in range(self.n):
            estimations[i] = np.random.random()
        return estimations

    def update_decision(self, id):

        # Set the decision each agent faces at time t, according to the fact he succeeded or not in his exchange at t-1,
        #  the decision he previously faced, and the choice he previously made.
        self.decision[id] = self.decision_array[self.finding_a_partner[id],
                                                self.decision[id],
                                                self.choice[id]]

    def update_options_values(self, id):

        # Each agent try to minimize the time to consume
        # That is v(option) = 1/(1/estimation)

        # Set value to each option choice

        self.value_ij[id] = self.estimation_ij[id]
        self.value_kj[id] = self.estimation_kj[id]

        if not (self.estimation_ik[id] + self.estimation_kj[id]) == 0:

            self.value_ik[id] = \
                (self.estimation_ik[id] * self.estimation_kj[id]) / \
                (self.estimation_ik[id] + self.estimation_kj[id])
        else:  # Avoid division by 0
            self.value_ik[id] = 0

        if not (self.estimation_ki[id] + self.estimation_ij[id]) == 0:
            self.value_ki[id] = \
                (self.estimation_ki[id] * self.estimation_ij[id]) / \
                (self.estimation_ki[id] + self.estimation_ij[id])
        else:  # Avoid division by 0
            self.value_ki[id] = 0

    def decision_rule(self, id):

        if self.decision[id] == 0:
            self.value_option0[id] = self.value_ij[id]
            self.value_option1[id] = self.value_ik[id]
        else:
            self.value_option0[id] = self.value_kj[id]
            self.value_option1[id] = self.value_ki[id]

        # id0 = np.where(self.decision == 0)[0]
        # id1 = np.where(self.decision == 1)[0]

        # self.value_option0[id0] = self.value_ij[id0]
        # self.value_option1[id0] = self.value_ik[id0]

        # self.value_option0[id1] = self.value_kj[id1]
        # self.value_option1[id1] = self.value_ki[id1]

        # Set a probability to current option 0 using softmax rule
        # (As there is only 2 options each time, computing probability for a unique option is sufficient)

        self.probability_of_choosing_option0[id] = \
            np.exp(self.value_option0[id] / self.temperature) / \
            (np.exp(self.value_option0[id] / self.temperature) +
             np.exp(self.value_option1[id] / self.temperature))

        self.random_number[id] = np.random.random()  # Generate random numbers

        # Make a choice using the probability of choosing option 0 and a random number for each agent
        # Choose option 1 if random number > or = to probability of choosing option 0,
        #  choose option 0 otherwise
        self.choice[id] = self.random_number[id] >= self.probability_of_choosing_option0[id]
        self.i_choice[id] = (self.decision[id] * 2) + self.choice[id]

        if self.i_choice[id] == 0:

            self.direct_exchange[str(self.type[id])] += 1

        else:

            self.indirect_exchange[str(self.type[id])] += 1

    def choice_transition_function(self, x, y):

        return self.choice_transition[x, y]

    def update_estimations(self, id):

        self.estimation_types = np.array(
            [self.estimation_ij[id], self.estimation_ik[id], self.estimation_kj[id], self.estimation_ki[id]])

        i_type = self.type[id]
        i_choice = self.i_choice[id]

        # Here, we take the type and the choice of agent i, in order to compare him with the other agents
        # and to compute his new estimation on the easiness to make the transaction he chose.
        # Throughout the rest, each array or list containing 4 elements will correspond to estimation_ij,
        # estimation_ik, estimation_kj and estimation_ki in this order.

        agents_types = list()
        agents_choices = list()
        agents_results = list()

        # We choose here a certain number (which corresponds to the quantity of information q_information)
        # of informers among all the agents

        informers = np.asarray(self.information)  # We get the agent's ids of all agent in the active agent vision,
        # then we retrieve their type, choice, and if they found a partner or not during the last trial

        agents_types = self.type[informers]
        agents_choices = self.i_choice[informers]
        agents_results = self.finding_a_partner[informers]

        relative_agents_type = (agents_types - i_type) % 3

        relative_choices = self.choice_transition_function(relative_agents_type, agents_choices)

        # print(relative_choices)

        averages = np.zeros(4)
        for j in range(len(averages)):
            id0 = np.where(np.asarray(relative_choices) == j)[0]
            if id0.size:

                averages[j] = np.mean(agents_results[id0])
            else:
                averages[j] = -1

            self.total_received_information[id] += len(id0)

        # Here, we have computed the right type of transactions from the point of view of agent i compared with
        # the type of choices which are made by the other agents. Then, once we have identified what agents
        # contributes to the 4 different estimations for i, we give their results corresponding to their
        # previous transaction according to the fact they succeeded in their transaction or not.

        my_opinion = np.zeros(4)

        my_opinion[i_choice] = self.epsilon * (self.finding_a_partner[id] - self.estimation_types[i_choice])

        others_opinion = np.zeros(4)
        for i in range(len(others_opinion)):
            if averages[i] is not -1:
                others_opinion[i] = (1 - self.epsilon) * (averages[i] - self.estimation_types[i])

        for i in range(len(others_opinion)):
            self.estimation_types[i] += self.alpha * (my_opinion[i] + others_opinion[i])

    def make_a_choice(self, id):

        self.update_decision(id)
        self.update_options_values(id)
        self.decision_rule(id)

    # ------------------------------------------------||| COMPUTE CHOICES PROPORTIONS |||---------------------------------------------------------------------------------------#

    def compute_choices_proportions(self):

        self.direct_choices_proportions = {"0": 0., "1": 0., "2": 0.}  # We reset direct choices proportions,
        # we compute it, to finally reset direct exchanges et indirect exchanges count

        for i in [0, 1, 2]:
            self.direct_choices_proportions[str(i)] = self.direct_exchange[str(i)] / (self.direct_exchange[str(i)] \
                                                                                      + self.indirect_exchange[str(i)])

        self.direct_exchange = {"0": 0., "1": 0., "2": 0.}
        self.indirect_exchange = {"0": 0., "1": 0., "2": 0.}

    def append_choices_to_compute_means(self):

        for i in [0, 1, 2]:
            self.choices_list[str(i)].append(self.direct_choices_proportions[str(i)])

    def compute_choices_means(self):

        list_mean = [np.mean(self.choices_list["0"]),
                     np.mean(self.choices_list["1"]),
                     np.mean(self.choices_list["2"])]

        return list_mean

    # ------------------------------------------------||| MONEY TEST |||--------------------------------------------------------------------------------------------------------#

    def test_for_money(self):

        money = -1

        # Money = 0?
        # type '0' should use direct exchange
        cond0 = np.mean(self.i_choice[self.idx0] == 0) > self.money_threshold

        # type '1' should use indirect exchange
        cond1 = np.mean([i in [1, 2] for i in self.i_choice[self.idx1]]) > self.money_threshold

        # type '2' should use direct exchange
        cond2 = np.mean(self.i_choice[self.idx2] == 0) > self.money_threshold

        if (cond0 * cond1 * cond2) == 1:

            money = 0

        else:

            # Money = 1?
            cond0 = np.mean(self.i_choice[self.idx0] == 0) > self.money_threshold
            cond1 = np.mean(self.i_choice[self.idx1] == 0) > self.money_threshold
            cond2 = np.mean([i in [1, 2] for i in self.i_choice[self.idx2]]) > self.money_threshold

            if (cond0 * cond1 * cond2) == 1:

                money = 1

            else:

                # Money = 2?
                cond0 = np.mean([i in [1, 2] for i in self.i_choice[self.idx0]]) > self.money_threshold
                cond1 = np.mean(self.i_choice[self.idx1] == 0) > self.money_threshold
                cond2 = np.mean(self.i_choice[self.idx2] == 0) > self.money_threshold

                if (cond0 * cond1 * cond2) == 1:
                    money = 2

        ##### END TEST #####
        ##### SAVE RESULT ####

        if money == -1:

            self.money_round[:] = 0
            self.money_time[:] = 0

        else:
            self.money_round[(money + 1) % 3] = 0
            self.money_round[(money + 2) % 3] = 0
            self.money_round[money] += 1
            # Memorize time for money emergence
            if self.money_round[money] == 1:
                self.money_time[money] = self.t

                # b = time()
                # self.dico['test_for_money'][0] += (b-a)
                # self.dico['test_for_money'][1] += 1

    def determine_money(self):

        for i in range(3):

            if self.money_round[i] >= self.money_delta:
                self.money = i
                self.t_emergence = self.money_time[i]
                break

                # b = time()
                # self.dico['determine_money'][0] += (b-a)
                # self.dico['determine_money'][1] += 1

    def compute_welfare(self, moment):

        # Compute rewards number
        d_exchange = np.where((self.i_choice == 0) * (self.finding_a_partner == 1))[0]
        i_exchange = np.where((self.i_choice == 2) * (self.finding_a_partner == 1))[0]

        if moment == 0:
            self.beginning_reward[self.t_beginning_reward, d_exchange] += 1
            self.beginning_reward[self.t_beginning_reward, i_exchange] += 1
            self.t_beginning_reward += 1

        else:

            self.end_reward[self.t_end_reward, d_exchange] += 1
            self.end_reward[self.t_end_reward, i_exchange] += 1
            self.t_end_reward += 1

    def summarise_welfare(self, moment):

        if moment == 0:

            reward = self.beginning_reward
            welfare = self.beginning_welfare

        else:
            reward = self.end_reward
            welfare = self.end_welfare

        a = np.zeros((self.welfare_delta, len(self.workforce)))

        for i in range(self.welfare_delta):
            a[i, 0] = np.mean(reward[i, self.idx0])
            a[i, 1] = np.mean(reward[i, self.idx1])
            a[i, 2] = np.mean(reward[i, self.idx2])

        welfare[0] = np.mean(a[:, 0])
        welfare[1] = np.mean(a[:, 1])
        welfare[2] = np.mean(a[:, 2])


# ---------------------------------------------||| MAIN RUNNER |||------------------------------------------------------------------------------------------------------------#
class SimulationRunner(object):
    def __init__(self):

        self.null = 0

        # self.map_limits = parameters["map_limits"]
        # # Create the economy to simulate
        # self.eco = Economy(parameters)

        # # Time simulation should last
        # self.t_max = parameters["tmax"]

        # result = self.main_runner()

    def main_runner(self, parameters):

        #

        self.map_limits = parameters["map_limits"]

        # Create the economy to simulate
        self.eco = Economy(parameters)

        # Time simulation should last
        self.t_max = parameters["tmax"]

        result = self.launch_economy()
        # self.stats()
        # self.save_data()
        return result

    def launch_economy(self):

        self.list_map = list()
        self.matrix_list0 = list()
        self.matrix_list1 = list()
        self.matrix_list2 = list()
        exchanges_proportions_list = list()

        for t in tqdm(range(self.t_max)):

            for id in range(self.eco.n):
                # move agent, then make them proceeding to exchange
                self.eco.move(id)
                self.eco.encounter(id)

                # create empty matrix
                matrix0 = np.zeros((self.map_limits["height"], self.map_limits["width"]))
                matrix1 = np.zeros((self.map_limits["height"], self.map_limits["width"]))
                matrix2 = np.zeros((self.map_limits["height"], self.map_limits["width"]))

                # copy positions dict (in order to use them during the graphic printing part)
                graphic_map = self.eco.graphic_map.copy()

                # fill the matrix with the exchanges positions
                matrix0[:] = self.eco.exchange_matrix["0"].copy()
                matrix1[:] = self.eco.exchange_matrix["1"].copy()
                matrix2[:] = self.eco.exchange_matrix["2"].copy()

                # For each "t" and each trial the matrix are added to a list
                self.matrix_list0.append(matrix0)
                self.matrix_list1.append(matrix1)
                self.matrix_list2.append(matrix2)

                # Same for graphics positions
                self.list_map.append(graphic_map)

            # for each "t" we compute the proportion of direct choices
            self.eco.compute_choices_proportions()

            # We append it to a list (in the fonction)
            self.eco.append_choices_to_compute_means()

            # We copy proportions and add them to a list
            proportions = self.eco.direct_choices_proportions.copy()
            exchanges_proportions_list.append(proportions)

        # Finally we compute the direct choices mean for each type
        # of agent and return it as well as the direct choices proportions

        list_mean = self.eco.compute_choices_means()
        result = {"exchanges_proportions_list": exchanges_proportions_list,
                  "list_mean": list_mean}
        return result

    def save_data(self):

        # Save matrix of exchanges and positions dict (in order to print the main map later)

        self.list_of_matrix = [self.matrix_list0, self.matrix_list1, self.matrix_list2]

        for i in range(len(self.list_of_matrix)):
            converter.write(self.list_of_matrix[i], table_name="exchange_{i}".format(i=i))

        pickle.dump(self.list_map, open("./data/map.p", mode='wb'))
        pickle.dump(self.exchanges_proportions_list, open("./data/exchanges.p", mode='wb'))

    def stats(self):

        # Prints direct choices proportions

        color_set = ["green", "blue", "red"]

        e = self.exchanges_proportions_list

        for j in [0, 1, 2]:
            plt.plot(np.arange(self.t_max), [e[k][str(j)] for k in range(len(e))], \
                     color=color_set[j], linewidth=1.0, label="Agent type = {j}".format(j=j))
        list_mean
        plt.suptitle('Direct choices proportion per type of agents', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', frameon=False)
        plt.show()


if __name__ == "__main__":

    map_limits = dict()
    map_limits["width"] = 25
    map_limits["height"] = 25

    parameters1 = \
        {
            "workforce": np.array([10, 10, 10], dtype=int),
            "alpha": 0.5,  # Set the coefficient learning
            "tau": 0.05,  # Set the softmax parameter.
            "time_limit": 1000,  # Set the number of time units the simulation will run
            "welfare_delta": 100,  # Set the number of time units you want for analysing welfare
            "money_threshold": .90,  # Set the 'money threshold
            "money_delta": 100,  # Set the number of time units to use for chec
            # if the economy is in a monetary state
            "stride": 1,  # by each agent at each round
            "epsilon": 0.5,
            "vision": 25,  # Set the importance of other agents'results in
            "area": 10,  # front of an individual res
            "map_limits": map_limits,
            "tmax": 1500

        }

    parameters2 = \
        {
            "workforce": np.array([10, 10, 10], dtype=int),
            "alpha": 0.5,  # Set the coefficient learning
            "tau": 0.05,  # Set the softmax parameter.
            "time_limit": 1000,  # Set the number of time units the simulation will run
            "welfare_delta": 100,  # Set the number of time units you want for analysing welfare
            "money_threshold": .90,  # Set the 'money threshold
            "money_delta": 100,  # Set the number of time units to use for chec
            # if the economy is in a monetary state
            "stride": 1,  # by each agent at each round
            "epsilon": 0.5,
            "vision": 15,  # Set the importance of other agents'results in
            "area": 10,  # front of an individual res
            "map_limits": map_limits,
            "tmax": 1500
        }

    parameters3 = \
        {
            "workforce": np.array([10, 10, 10], dtype=int),
            "alpha": 0.5,  # Set the coefficient learning
            "tau": 0.05,  # Set the softmax parameter.
            "time_limit": 1000,  # Set the number of time units the simulation will run
            "welfare_delta": 100,  # Set the number of time units you want for analysing welfare
            "money_threshold": .90,  # Set the 'money threshold
            "money_delta": 100,  # Set the number of time units to use for chec
            # if the economy is in a monetary state
            "stride": 1,  # by each agent at each round
            "epsilon": 0.5,
            "vision": 10,  # Set the importance of other agents'results in
            "area": 10,  # front of an individual res
            "map_limits": map_limits,
            "tmax": 1500
        }

    parameters4 = \
        {
            "workforce": np.array([10, 10, 10], dtype=int),
            "alpha": 0.5,  # Set the coefficient learning
            "tau": 0.05,  # Set the softmax parameter.
            "time_limit": 1000,  # Set the number of time units the simulation will run
            "welfare_delta": 100,  # Set the number of time units you want for analysing welfare
            "money_threshold": .90,  # Set the 'money threshold
            "money_delta": 100,  # Set the number of time units to use for chec
            # if the economy is in a monetary state
            "stride": 1,  # by each agent at each round
            "epsilon": 0.5,
            "vision": 5,  # Set the importance of other agents'results in
            "area": 10,  # front of an individual res
            "map_limits": map_limits,
            "tmax": 1500
        }

    S = SimulationRunner()
    p = Pool(processes=4)
    result = p.map(S.main_runner, [parameters1, parameters2, parameters3, parameters4])

    list_0 = list()
    list_1 = list()

    for i in range(len(result)):
        list_0.append(result[i]["exchanges_proportions_list"])
        list_1.append(result[i]["list_mean"])

    pickle.dump(list_0, open("./data/global_list.p", mode="wb"))
    pickle.dump(list_1, open("./data/means.p", mode="wb"))