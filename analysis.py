import matplotlib.pyplot as plt 
import numpy as np
import pickle
from tqdm import tqdm


# # ------------------------------------------------||| MONEY TEST |||--------------------------------------------------------------------------------------------------------#
#
# def test_for_money(self):
#     money = -1
#
#     # Money = 0?
#     # type '0' should use direct exchange
#     cond0 = np.mean(self.i_choice[self.idx0] == 0) > self.money_threshold
#
#     # type '1' should use indirect exchange
#     cond1 = np.mean([i in [1, 2] for i in self.i_choice[self.idx1]]) > self.money_threshold
#
#     # type '2' should use direct exchange
#     cond2 = np.mean(self.i_choice[self.idx2] == 0) > self.money_threshold
#
#     if (cond0 * cond1 * cond2) == 1:
#
#         money = 0
#
#     else:
#
#         # Money = 1?
#         cond0 = np.mean(self.i_choice[self.idx0] == 0) > self.money_threshold
#         cond1 = np.mean(self.i_choice[self.idx1] == 0) > self.money_threshold
#         cond2 = np.mean([i in [1, 2] for i in self.i_choice[self.idx2]]) > self.money_threshold
#
#         if (cond0 * cond1 * cond2) == 1:
#
#             money = 1
#
#         else:
#
#             # Money = 2?
#             cond0 = np.mean([i in [1, 2] for i in self.i_choice[self.idx0]]) > self.money_threshold
#             cond1 = np.mean(self.i_choice[self.idx1] == 0) > self.money_threshold
#             cond2 = np.mean(self.i_choice[self.idx2] == 0) > self.money_threshold
#
#             if (cond0 * cond1 * cond2) == 1:
#                 money = 2
#
#     ##### END TEST #####
#     ##### SAVE RESULT ####
#
#     if money == -1:
#
#         self.money_round[:] = 0
#         self.money_time[:] = 0
#
#     else:
#         self.money_round[(money + 1) % 3] = 0
#         self.money_round[(money + 2) % 3] = 0
#         self.money_round[money] += 1
#         # Memorize time for money emergence
#         if self.money_round[money] == 1:
#             self.money_time[money] = self.t
#
#             # b = time()
#             # self.dico['test_for_money'][0] += (b-a)
#             # self.dico['test_for_money'][1] += 1
#
#
# def determine_money(self):
#     for i in range(3):
#
#         if self.money_round[i] >= self.money_delta:
#             self.money = i
#             self.t_emergence = self.money_time[i]
#             break
#
#             # b = time()
#             # self.dico['determine_money'][0] += (b-a)
#             # self.dico['determine_money'][1] += 1
#
#
# def compute_welfare(self, moment):
#     # Compute rewards number
#     d_exchange = np.where((self.i_choice == 0) * (self.finding_a_partner == 1))[0]
#     i_exchange = np.where((self.i_choice == 2) * (self.finding_a_partner == 1))[0]
#
#     if moment == 0:
#         self.beginning_reward[self.t_beginning_reward, d_exchange] += 1
#         self.beginning_reward[self.t_beginning_reward, i_exchange] += 1
#         self.t_beginning_reward += 1
#
#     else:
#
#         self.end_reward[self.t_end_reward, d_exchange] += 1
#         self.end_reward[self.t_end_reward, i_exchange] += 1
#         self.t_end_reward += 1
#
#
# def summarise_welfare(self, moment):
#     if moment == 0:
#
#         reward = self.beginning_reward
#         welfare = self.beginning_welfare
#
#     else:
#         reward = self.end_reward
#         welfare = self.end_welfare
#
#     a = np.zeros((self.welfare_delta, len(self.workforce)))
#
#     for i in range(self.welfare_delta):
#         a[i, 0] = np.mean(reward[i, self.idx0])
#         a[i, 1] = np.mean(reward[i, self.idx1])
#         a[i, 2] = np.mean(reward[i, self.idx2])
#
#     welfare[0] = np.mean(a[:, 0])
#     welfare[1] = np.mean(a[:, 1])
#     welfare[2] = np.mean(a[:, 2])

last = open("../data/last.txt").read()

parameters = pickle.load(open("../data/parameters_{}.p".format(last), mode="rb"))

data = pickle.load(open("../data/exchanges_{}.p".format(last), mode="rb"))

t_max = len(data)

agents_proportions = np.zeros((t_max, 3))

for i in range(t_max):

    for j in range(3):

        agents_proportions[i, j] = data[i][str(j)]

color_set = ["green", "blue", "red"]

for agent_type in range(3):

    plt.plot(np.arange(t_max), agents_proportions[:, agent_type],
             color=color_set[agent_type], linewidth=1.0)
#
# plt.suptitle('Direct choices proportion per type of agents', fontsize=14, fontweight='bold')
# plt.legend(loc='lower left', frameon=False)
    plt.ylim([0, 1])
#
#
plt.show()
# plt.savefig("figure.pdf")
#
# plt.close()
