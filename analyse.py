import pickle

import analysis.separate
import analysis.summary
import analysis.dynamic


def main(file_name):

    with open("data/pickle/{}.p".format(file_name), 'rb') as f:
        results_pool = pickle.load(f)

    # analysis.separate.plot_indirect_exchanges(data=results_pool)
    # analysis.summary.plot(data=results_pool)
    analysis.dynamic.plot_moves(data=results_pool, number=4)
    # analysis.dynamic.plot_exchanges(data=results_pool, number=4)


if __name__ == "__main__":
    # f_name = "pool_18_02_26_16_16_05_748118"
    f_name = "pool_18_03_19_21_25_58_613134"
    main(f_name)
