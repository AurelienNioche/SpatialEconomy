import numpy as np
import itertools


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
# * '1' means a type-23 agent;
# * '2' means a type-31 agent.

# For a decision:
# * '0' means 'type-i decision' (agent has 'i' in hand);
# * '1' means 'type-k decision' (agent has '' in hand).

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


class Model:

    def __init__(self, vision_area=5, movement_area=5, stride=1, x0=10, x1=10, x2=10,
                 alpha=0.1, tau=0.05, map_width=20, map_height=2):

        # Get parameters

        self.vision_area = vision_area

        self.movement_area = movement_area
        self.stride = stride

        self.map_width = map_width
        self.map_height = map_height

        self.workforce = np.array([x0, x1, x2], dtype=int)  # Number of agents by type

        self.alpha = alpha  # Learning coefficient
        self.temperature = tau  # Softmax parameter

        # Attributes for computation

        self.n = sum(self.workforce)  # Total number of agents

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
        self.int_to_relative_choice = np.array([[0, 1, -1, -1, 3, 2],
                                                [3, 2, 1, 0, -1, -1],
                                                [-1, -1, 2, 3, 0, 1]], dtype=int)

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

        self.type = np.zeros(self.n, dtype=int)
        self.good = np.zeros(self.n, dtype=int)

        self.type[:] = np.array(
            [0, ] * self.workforce[0] + [1, ] * self.workforce[1] + [2, ] * self.workforce[2])
        self.good[:] = np.array(
            [0, ] * self.workforce[0] + [1, ] * self.workforce[1] + [2, ] * self.workforce[2])

        # Each agent possesses an index by which he can be identified.
        #  Here are the the indexes lists corresponding to each type of agent:
        idx = np.arange(self.n)

        self.idx0 = idx[self.type == 0]
        self.idx1 = idx[self.type == 1]
        self.idx2 = idx[self.type == 2]

        self.position = np.zeros((self.n, 2), dtype=int)
        self.position[:] = -1

        self.x_perimeter = np.zeros((self.n, 2))
        self.y_perimeter = np.zeros((self.n, 2))

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
        self.estimation = np.zeros((self.n, 4))  # There is 4 exchange types relevant for each agent

        # For each cell, it will contain the number of exchange for every of the three goods
        self.exchange_map = np.zeros(shape=(self.map_width, self.map_height, 3), dtype=int)

        # For each cell, it will contain idx of agent
        self.agent_map = np.zeros((self.map_width, self.map_height), dtype=int)

        self.direct_choices_proportions = np.zeros(3)
        self.indirect_choices_proportions = np.zeros(3)

        self.direct_exchange = np.zeros(3)
        self.indirect_exchange = np.zeros(3)

        self.exchange_counter = np.zeros(3)

        # This is the initial guest (same for every agent).
        # '1' means each type of exchange can be expected to be realized in only one unit of time
        # The more the value is close to zero, the more an exchange is expected to be hard.
        #
        self.estimation[:] = np.random.random((self.n, 4))

        self.setup()


# --------------------------------------------------||| SETUP |||----------------------------------------------- #

    def setup(self):

        self.setup_insert_agents_on_map()
        self.setup_define_perimeters()

    def setup_insert_agents_on_map(self):

        croissant_order = range(self.n)
        random_order = np.random.permutation(croissant_order)

        for idx in random_order:

            while True:
                x = np.random.randint(0, self.map_width)
                y = np.random.randint(0, self.map_height)

                if self.agent_map[x, y] != -1:
                    self.position[idx] = x, y
                    self.agent_map[x, y] = idx

                    break

    def setup_define_perimeters(self):

        self.x_perimeter[:, 0] = self.position[:, 0] - self.movement_area
        self.x_perimeter[:, 1] = self.position[:, 0] + self.movement_area
        self.y_perimeter[:, 0] = self.position[:, 1] - self.movement_area
        self.y_perimeter[:, 1] = self.position[:, 1] + self.movement_area

    # --------------------------------------------------||| RESET |||----------------------------------------------- #

    def reset(self):
        self.direct_exchange[:] = 0.
        self.indirect_exchange[:] = 0.
        self.exchange_counter[:] = 0.

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
                                        position[1] + i[1]) for i in itertools.product([-1, 0, 1], repeat=2)])

        # nearby_positions is a matrix of 9*2 (9: number of cases around the agent, included his own position,
        # 2: x, y coordinates)

        # We look at x  and y columns to check that they are in map dimensions
        result_x_inf = nearby_positions[:, 0] < self.map_width
        result_y_inf = nearby_positions[:, 1] < self.map_height

        result_x_sup = nearby_positions[:, 0] >= 0
        result_y_sup = nearby_positions[:, 1] >= 0

        b_positions_in_map_x = result_x_inf * result_x_sup
        b_positions_in_map_y = result_y_inf * result_y_sup

        b_positions_in_map = b_positions_in_map_x * b_positions_in_map_y

        positions_in_map = nearby_positions[b_positions_in_map]

        # test if they are in the perimeter of the agent

        result_x_inf = positions_in_map[:, 0] < self.x_perimeter[idx, 1]  # 1: max
        result_x_sup = positions_in_map[:, 0] >= self.x_perimeter[idx, 0]  # 0: min

        result_y_inf = positions_in_map[:, 1] < self.y_perimeter[idx, 1]
        result_y_sup = positions_in_map[:, 1] >= self.y_perimeter[idx, 0]

        b_positions_in_map_x = result_x_inf * result_x_sup
        b_positions_in_map_y = result_y_inf * result_y_sup

        b_positions_in_map = b_positions_in_map_x * b_positions_in_map_y

        return positions_in_map[b_positions_in_map]

    def move_find_free_position(self, idx, positions_in_map):

        assert type(idx) in (np.int64, int)

        np.random.shuffle(positions_in_map)

        for x, y in positions_in_map:

            # If position is available
            if self.agent_map[x, y] == -1:
                
                # Agent takes i as new position
                self.position[idx] = 0
                self.agent_map[x, y] = idx
                break

# ------------------------------------------------||| MAKE ENCOUNTER |||--------------------------------------- #

    def encounter(self, idx):

        assert type(idx) in (np.int64, int)

        print("Encounter:", idx, "\n")

        occupied_nearby_positions = self.encounter_check_nearby_positions(idx)
        group_idx = self.encounter_look_for_partners(occupied_nearby_positions)

        choice_current_agent, proportion_of_matching_choices, partner_id = \
            self.encounter_look_for_partners_choices(idx, group_idx)

        self.encounter_update_estimations(idx=idx, group_idx=group_idx,
                                          acceptance_frequency=proportion_of_matching_choices,
                                          exchange_type=choice_current_agent)
        if partner_id != -1:
            self.encounter_proceed_to_exchange(idx, partner_id)

    def encounter_check_nearby_positions(self, idx):

        assert type(idx) in (np.int64, int)
        
        x, y = self.position[idx]
        
        nearby_positions_x = [
            x - self.vision_area,
            x + self.vision_area
        ]
        nearby_positions_y = [
            y - self.vision_area,
            y + self.vision_area
        ]

        p = self.position

        result_x_inf = p[:, 0] <= nearby_positions_x[1]
        result_y_inf = p[:, 1] <= nearby_positions_y[1]
        result_x_sup = p[:, 0] >= nearby_positions_x[0]
        result_y_sup = p[:, 1] >= nearby_positions_y[0]

        b_occupied_nearby_positions_x = result_x_inf * result_x_sup
        b_occupied_nearby_positions_y = result_y_inf * result_y_sup

        b_occupied_nearby_positions = b_occupied_nearby_positions_x * b_occupied_nearby_positions_y

        occupied_nearby_positions = p[b_occupied_nearby_positions]

        return occupied_nearby_positions

    def encounter_look_for_partners(self, positions_in_map):

        idx_informers = []
        for x, y in positions_in_map:
            idx_informers.append(self.agent_map[x, y])
            
        return idx_informers

    def encounter_look_for_partners_choices(self, idx, group_idx):

        print("encounter_look_for_partners_choices:", idx, type(idx))
        assert type(idx) in (np.int64, int)

        # The agent chooses the good he wants to obtain and asks agents around him for it

        self.choose(idx)
        choice_current_agent = list(self.absolute_matrix[self.i_choice[idx], self.type[idx]])
        int_choice_current_agent = self.absolute_exchange_to_int[tuple(choice_current_agent)]

        matching_list = list()

        # We retrieve the good wanted by the others and check if their needs and our agent need match

        partner_ids = []

        for partner_id in group_idx:

            assert type(partner_id) in (int, np.int64)

            self.choose(partner_id)
            choice_current_partner = list(self.absolute_matrix[self.i_choice[partner_id], self.type[partner_id]])
            success = choice_current_partner[::-1] == choice_current_agent
            matching_list.append(success)
            if success:
                partner_ids.append(partner_id)

        if partner_ids:

            partner_id = np.random.choice(partner_ids)
        else:
            partner_id = -1  # Partner_id must be an int, therefore we give it an unlikely
            # value in case the agent doesn't have a partner

        proportion_of_matching_choices = np.mean(matching_list)

        return int_choice_current_agent, proportion_of_matching_choices, partner_id

    def encounter_proceed_to_exchange(self, idx, partner_id):

        assert type(idx) in (np.int64, int)

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
        # 
        # self.position_map[self.position[idx]] = self.type[idx], self.good[idx]
        # self.position_map[self.position[partner_id]] = self.type[partner_id], self.good[partner_id]

    def encounter_update_estimations(self, idx, group_idx, acceptance_frequency, exchange_type):

        assert type(idx) in (np.int64, int)

        group_in_large_sense = group_idx + [idx]
        for idx in group_in_large_sense:
            relative_choice = self.int_to_relative_choice[self.type[idx], exchange_type]

            self.estimation[idx, relative_choice] += \
                self.alpha * (acceptance_frequency - self.estimation[idx, relative_choice])

    def encounter_exchange_count(self, idx, partner_id):

        assert type(idx) in (np.int64, int)

        x, y = self.position[partner_id]

        if self.good[idx] + self.good[partner_id] == 1:
            self.exchange_map[x, y, 0] += 1

        elif self.good[idx] + self.good[partner_id] == 3:
            self.exchange_map[x, y, 1] += 1

        else:
            self.exchange_map[x, y, 2] += 1

# ----------------------------------------------------||| CHOICE ||| -------------------------------------------- #

    def choose(self, idx):

        print("Choose", idx, type(idx))

        assert type(idx) in (np.int64, int)

        self.choose_update_options_values(idx)
        self.choose_decision_rule(idx)

    def choose_update_options_values(self, idx):

        # Each agent try to minimize the time to consume
        # That is v(option) = 1/(1/estimation)

        # Set value to each option choice

        self.value_ij[idx] = self.estimation[idx, 0]
        self.value_kj[idx] = self.estimation[idx, 0]

        print("choose_update_options_values", type(idx), idx)
        print(self.estimation.shape)

        print(self.estimation[idx, 1] + self.estimation[idx, 2])
        if not self.estimation[idx, 1] + self.estimation[idx, 2] == 0:

            self.value_ik[idx] = \
                (self.estimation[idx, 1] * self.estimation[idx, 2]) / \
                (self.estimation[idx, 1] + self.estimation[idx, 2])
        else:  # Avoid division by 0
            self.value_ik[idx] = 0

        if not (self.estimation[idx, 3] + self.estimation[idx, 0]) == 0:
            self.value_ki[idx] = \
                (self.estimation[idx, 3] * self.estimation[idx, 0]) / \
                (self.estimation[idx, 3] + self.estimation[idx, 0])
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
            1 / \
            (1 + np.exp(- (self.value_option0[idx] - self.value_option1[idx]) / self.temperature))

        random_number = np.random.random()  # Generate random number

        # Make a choice using the probability of choosing option 0 and a random number for each agent
        # Choose option 1 if random number > or = to probability of choosing option 0,
        #  choose option 0 otherwise
        self.choice[idx] = random_number >= probability_of_choosing_option0
        self.i_choice[idx] = (self.decision[idx] * 2) + self.choice[idx]

        self.exchange_counter[self.type[idx]] += 1

        if self.i_choice[idx] == 0:

            self.direct_exchange[self.type[idx]] += 1

        elif self.i_choice[idx] in [1, 2]:

            self.indirect_exchange[self.type[idx]] += 1


# ------------------------------------------------||| COMPUTE CHOICES PROPORTIONS |||---------------------------- #

    def compute_choices_proportions(self):

        if self.exchange_counter.all() > 0:

            self.direct_choices_proportions[:] = \
                self.direct_exchange[:] / self.exchange_counter[:]

            self.indirect_choices_proportions[:] = \
                self.indirect_exchange[:] / self.exchange_counter[:]
        else:

            for i in range(3):

                if self.exchange_counter[i] > 0:
                    self.direct_choices_proportions[i] = \
                        self.direct_exchange[i] / self.exchange_counter[i]
                    self.indirect_choices_proportions[i] = \
                        self.indirect_exchange[i] / self.exchange_counter[i]
                else:
                    self.direct_choices_proportions[i] = 0
                    self.indirect_choices_proportions[i] = 0


def main():

    m = Model()



if __name__ == "__main__":

    main()
