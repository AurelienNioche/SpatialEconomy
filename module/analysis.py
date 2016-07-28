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

   #  @classmethod
    # def import_data(cls, suffix):

        # parameters = pickle.load(open("../data/parameters/parameters_{}.p".format(suffix), mode="rb"))
        # print(parameters)

        # direct_exchange = pickle.load(open("../data/exchanges/direct_exchanges_{}.p".format(suffix), mode="rb"))
        # indirect_exchange = pickle.load(open("../data/exchanges/ indirect_exchanges_{}.p".format(suffix), mode="rb"))

        # return parameters, direct_exchange, indirect_exchange

    def analyse(self, suffix):

        parameters, direct_exchange, indirect_exchange = self.import_data(suffix=suffix)
        money_timeline = np.zeros(parameters["t_max"])
        money = {0: 0, 1: 0, 2: 0, -1: 0}
        interruptions = 0 

        for t in range(parameters["t_max"]):

            money_t = self.test_for_money_state(direct_exchange=direct_exchange[t],
                                                indirect_exchange=indirect_exchange[t])
            money_timeline[t] = money_t  
            money[money_t] += 1

            cond0 = money_t == -1  
            cond1 = money_timeline[t-1] != -1
            
            interruptions += cond0 * cond1    
         

        results = \
                {"money" : money, 
                 "interruptions": interruptions, 
                 "money_timeline": money_timeline}
        
        return results, suffix
    
    def save_data(self, results, suffix):

        pickle.dump(results, open("../data/results_money_test_{}.p".format(suffix), mode='wb'))
        

# ------------------------------------------------||| MAIN  |||----------------------------------------------- #   



def main(suffix):

    m = MoneyAnalysis()
    results = m.analyse(suffix)
    m.save_data(results) 
    
if __name__ == "__main__":

    main(suffix="bhablfbrbh")
