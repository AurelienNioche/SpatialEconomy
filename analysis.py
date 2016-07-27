import matplotlib.pyplot as plt 
import numpy as np
import pickle
from tqdm import tqdm


# ------------------------------------------------||| MONEY TEST |||----------------------------------------------- #

class MoneyAnalysis(object):

    def __init__(self):

        self.money_threshold = .90

    def test_for_money_state(self, direct_exchange, indirect_exchange):

        money = -1

        # Money = 0?
        # type '0' should use direct exchange
        cond0 = direct_exchange[0] > self.money_threshold

        # type '1' should use indirect exchange
        cond1 = indirect_exchange[1] > self.money_threshold

        # type '2' should use direct exchange
        cond2 = direct_exchange[2] > self.money_threshold

        if (cond0 * cond1 * cond2) == 1:

            money = 0

        else:

            # Money = 1?
            cond0 = direct_exchange[0] > self.money_threshold
            cond1 = direct_exchange[1] > self.money_threshold
            cond2 = indirect_exchange[2] > self.money_threshold

            if (cond0 * cond1 * cond2) == 1:

                money = 1

            else:

                # Money = 2?
                cond0 = indirect_exchange[0] > self.money_threshold
                cond1 = direct_exchange[1] > self.money_threshold
                cond2 = direct_exchange[2] > self.money_threshold

                if (cond0 * cond1 * cond2) == 1:
                    money = 2

        return money

    @classmethod
    def import_data(cls, suffix):

        parameters = pickle.load(open("../data/parameters_{}.p".format(suffix), mode="rb"))
        print(parameters)

        direct_exchange = pickle.load(open("../data/direct_exchanges_{}.p".format(suffix), mode="rb"))
        indirect_exchange = pickle.load(open("../data/indirect_exchanges_{}.p".format(suffix), mode="rb"))

        return parameters, direct_exchange, indirect_exchange

    def analyse(self, suffix):

        parameters, direct_exchange, indirect_exchange = self.import_data(suffix=suffix)
        money_timeline = np.zeros(parameters["t_max"])
        money = {"0": 0, "1": 0, "2": 0}
        no_money = 0
        interruptions = 0 

        for t in range(parameters["t_max"]):

            money_t = self.test_for_money_state(direct_exchange=direct_exchange[t],
                                                indirect_exchange=indirect_exchange[t])
            money_timeline[t] = money_t  

            if money_t == 0:
                money["0"] += 1
            
            elif money_t == 1:
                money["1"] += 1
            
            elif money_t == 2:
                money["2"] += 1
            
            else: 
                no_money += 1
            
            if money_t == -1 and money_timeline[t-1] != -1:
                
                interruptions += 1

        return money, no_money, interruptions, money_timeline

class GraphProportionChoices(object):

    def __init__(self):

        pass

    @classmethod
    def plot(cls, suffix):
        
        parameters = pickle.load(open("../data/parameters_{}.p".format(suffix), mode="rb"))
        print(parameters)

        data = pickle.load(open("../data/exchanges_{}.p".format(suffix), mode="rb"))

        t_max = len(data)

        agents_proportions = np.zeros((t_max, 3))

        for i in range(t_max):

            for j in range(3):
                agents_proportions[i, j] = data[i][str(j)]

        color_set = ["green", "blue", "red"]

        for agent_type in range(3):
            plt.plot(np.arange(t_max), agents_proportions[:, agent_type],
                     color=color_set[agent_type], linewidth=1.0)
            
            plt.ylim([0, 1])

        # plt.suptitle('Direct choices proportion per type of agents', fontsize=14, fontweight='bold')
        # plt.legend(loc='lower left', frameon=False)

        # plt.savefig("figure.pdf")
        plt.show()


def main(suffix):

    m = MoneyAnalysis()
    m.analyse(suffix)

if __name__ == "__main__":

    main(suffix="bhablfbrbh")
