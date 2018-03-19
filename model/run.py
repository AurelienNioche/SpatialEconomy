import numpy as np
import tqdm

from . import model, data_structure


def run(t_max=600, map_height=30, map_width=30,
        alpha=0.4, tau=0.01, movement_area=6, vision_area=15,
        x0=65, x1=65, x2=65, stride=1, seed=np.random.randint(0, 2**32-1),
        graphics=False, multi=False):

    np.random.seed(seed)

    # tqdm.tqdm_gui.write("Producing data...")

    eco = model.Model(
        map_height=map_height, map_width=map_width,
        x0=x0, x1=x1, x2=x2,
        vision_area=vision_area, movement_area=movement_area, stride=stride,
        alpha=alpha, tau=tau
    )

    direct_exchanges_proportions = np.zeros((t_max, 3))
    indirect_exchanges_proportions = np.zeros((t_max, 3))

    idx = np.arange(eco.n, dtype=int)

    agent_maps = None
    exchange_maps = None

    # Place agents and stuff...
    eco.setup()

    if graphics:

        agent_maps = np.zeros((t_max, map_width, map_height), dtype=int)
        exchange_maps = np.zeros((t_max, 3, map_width, map_height), dtype=int)

        # Save initial positions
        agent_maps[0] = eco.agent_map

    if multi:
        iterable = range(t_max)
    else:
        iterable = tqdm.tqdm(range(t_max))

    for t in iterable:

        eco.reset()

        np.random.shuffle(idx)

        for i in idx:

            # move agent, then make them proceeding to exchange
            if stride > 0:
                eco.move(i)
            eco.encounter(i)

        # -------------------- #
        # For saving...
        # ------------------- #
        if graphics:

            agent_maps[t] = eco.agent_map
            exchange_maps[t] = eco.exchange_map

        # -----------------  #

        # ---------- #
        # Do some stats...

        # for each "t" we compute the proportion of direct choices
        eco.compute_choices_proportions()

        direct_exchanges_proportions[t, :] = eco.direct_choices_proportions
        indirect_exchanges_proportions[t, :] = eco.indirect_choices_proportions

    # Finally we compute the direct choices mean for each type
    # of agent and return it as well as the direct choices proportions

    # tqdm.tqdm_gui.write("\nDone!")

    parameters = data_structure.Parameters(
        t_max=t_max, map_height=map_height, map_width=map_width,
        x0=x0, x1=x1, x2=x2,
        vision_area=vision_area, movement_area=movement_area, stride=stride,
        alpha=alpha, tau=tau, seed=seed, graphics=graphics
    )

    return data_structure.Result(
        direct_exchanges_proportions=direct_exchanges_proportions,
        indirect_exchanges_proportions=indirect_exchanges_proportions,
        exchange_maps=exchange_maps, agent_maps=agent_maps,
        parameters=parameters
    )
