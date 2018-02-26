import multiprocessing
import tqdm
import numpy as np
import os
import shutil
import json

import model
import analysis

parameters_folder = "parameters"
parameters_files = {
    "single": "{}/parameters_single.json".format(parameters_folder),
    "pool": "{}/parameters_pool.json".format(parameters_folder)
}


def run(kwargs):
    return model.run(**kwargs)


def prepare():

    for v in parameters_files.values():
        if not os.path.exists(v):
            os.makedirs(os.path.dirname(v), exist_ok=True)
            shutil.rmtree("parameters")
            shutil.copytree("template", "parameters")
            break


def main_multi():

    with open(parameters_files["pool"], "r") as f:
        pp = model.data_structure.ParametersPool(**json.load(f))

    np.random.seed(pp.seed)

    parameters_list = []

    seeds = np.random.randint(0, 2**32-1, size=pp.n)

    for i in range(pp.n):

        x = np.random.randint(pp.x_min, pp.x_max + 1)

        parameters_list.append(
            model.data_structure.Parameters(
                x0=x,
                x1=x,
                x2=x,
                stride=np.random.randint(pp.stride_min, pp.stride_max + 1),
                movement_area=np.random.randint(
                    pp.movement_area_min, pp.movement_area_max),
                vision_area=np.random.randint(
                    pp.vision_area_min, pp.vision_area_max + 1
                ),
                alpha=np.random.uniform(
                  pp.alpha_min, pp.alpha_max
                ),
                tau=np.random.uniform(
                    pp.tau_min, pp.tau_max
                ),
                map_width=pp.map_width,
                map_height=pp.map_height,
                t_max=pp.t_max,
                seed=seeds[i]
            ).__dict__
        )

    pool = multiprocessing.Pool()

    backups = []

    for bkp in tqdm.tqdm(
            pool.imap_unordered(run, parameters_list),
            total=pp.n):
        backups.append(bkp)

    r = model.data_structure.ResultPool(data=backups, parameters=pp)
    r.save()

    analysis.pool.run()


def main_single():

    with open(parameters_files["single"], "r") as f:
        parameters = json.load(f)

    r = run(parameters)
    r.save()

    analysis.separate.run(r)


if __name__ == "__main__":

    prepare()
    # main_single()
    main_multi()