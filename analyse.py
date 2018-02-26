import numpy as np
import os
import shutil
import json
import pickle

import model
import analysis


def main(file_name):

    with open("data/pickle/{}.p".format(file_name), 'rb') as f:
        results_pool = pickle.load(f)
    analysis.pool.run(results_pool=results_pool)


if __name__ == "__main__":
    f_name = "pool_18_02_26_16_16_05_748118"
    main(f_name)
