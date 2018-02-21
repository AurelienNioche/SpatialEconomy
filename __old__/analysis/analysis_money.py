import numpy as np
from tqdm import tqdm
from sqlite3 import connect
from os import path
from analysis.import_data import DataImporter


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

        return {
            "m0": money[0],
            "m1": money[1],
            "m2": money[2],
            "m_sum": money[0] + money[1] + money[2],
            "interruptions": interruptions
        }


# ------------------------------------------------||| BACKUP  |||----------------------------------------------- #


class BackUp(object):

    def __init__(self, session_suffix):

        self.connection = connect("../../results/results_{}.db".format(session_suffix))
        self.cursor = self.connection.cursor()

        self.idx = 0

    def create_table(self):

        query = "CREATE TABLE `results` " \
                "(`ID` INTEGER PRIMARY KEY, " \
                "`eco_idx` INTEGER," \
                " `vision_area` INTEGER, " \
                "`movement_area` INTEGER, `stride` INTEGER, " \
                "`width` INTEGER, `height` INTEGER , " \
                "`x0` INTEGER, `x1` INTEGER, `x2` INTEGER, " \
                "`alpha` REAL, `tau` REAL, `t_max` INTEGER," \
                "`m0` INTEGER," \
                "`m1` INTEGER," \
                "`m2` INTEGER," \
                "`m_sum` INTEGER," \
                "`interruptions` INTEGER);"

        self.cursor.execute(query)

    def save(self, eco_idx, parameters, results):

        query = "INSERT INTO `results` " \
                "(`ID`, " \
                "`eco_idx`," \
                " `vision_area`, " \
                "`movement_area`, " \
                "`stride`, " \
                "`width`, " \
                "`height`, " \
                "`x0`, `x1`, `x2`, " \
                "`alpha`, `tau`, `t_max`," \
                "`m0`, `m1`, `m2`," \
                "`m_sum`," \
                "`interruptions`) " \
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"

        self.cursor.execute(
            query, (
                self.idx,
                int(eco_idx),
                parameters["vision_area"],
                parameters["movement_area"],
                parameters["stride"],
                parameters["width"],
                parameters["height"],
                parameters["x0"],
                parameters["x1"],
                parameters["x2"],
                parameters["alpha"],
                parameters["tau"],
                parameters["t_max"],
                results["m0"],
                results["m1"],
                results["m2"],
                results["m_sum"],
                int(results["interruptions"])
            )
        )
        self.idx += 1

    def close(self):

        self.connection.commit()
        self.connection.close()


# ------------------------------------------------||| MAIN  |||----------------------------------------------- #


def main():

    session_suffix = "2016_11_17"

    data_importer = DataImporter(session_suffix=session_suffix)
    back_up = BackUp(session_suffix=session_suffix)
    money_analysis = MoneyAnalysis()

    back_up.create_table()

    if not path.exists("../../temp/all_eco_idx.npy"):
        print("Get eco indexes from db.")
        all_eco_idx = data_importer.get_eco_idx()
        np.save("../../temp/all_eco_idx.npy", all_eco_idx)
    else:
        print("Get eco indexes from npy file.")
        all_eco_idx = np.load("../../temp/all_eco_idx.npy")

    print("Number of economies:", len(all_eco_idx))

    for eco_idx in tqdm(all_eco_idx):

        parameters = data_importer.get_parameters(eco_idx)
        direct_exchange = data_importer.get_exchanges(eco_idx=eco_idx, type_of_exchange="direct")
        indirect_exchange = data_importer.get_exchanges(eco_idx=eco_idx, type_of_exchange="indirect")

        results = money_analysis.analyse(
            direct_exchange=direct_exchange,
            indirect_exchange=indirect_exchange,
            t_max=parameters["t_max"])

        back_up.save(eco_idx=eco_idx, parameters=parameters, results=results)

    data_importer.close()
    back_up.close()


if __name__ == "__main__":

    main()
