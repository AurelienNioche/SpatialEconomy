import numpy as np


class MoneyAnalysis:

    def __init__(self, m0, m1, m2, interruptions):

        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.interruptions = interruptions


class MoneyAnalyst(object):

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

    def analyse(self, direct_exchange, indirect_exchange, t_max):

        money_timeline = np.zeros(t_max)
        money = {0: 0, 1: 0, 2: 0, -1: 0}
        interruptions = 0

        for t in range(t_max):

            money_t = self.test_for_money_state(direct_exchange=direct_exchange[t],
                                                indirect_exchange=indirect_exchange[t])
            money_timeline[t] = money_t
            money[money_t] += 1

            if t > 0:

                cond0 = money_t == -1
                cond1 = money_timeline[t-1] != -1
                interruptions += cond0 * cond1

        return MoneyAnalysis(
            m0=money[0], m1=money[1], m2=money[2],
            interruptions=interruptions)


def run():
    pass
