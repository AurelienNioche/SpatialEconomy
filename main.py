import tqdm
import numpy as np

import model


class Result:

    def __init__(self, direct_exchanges_proportions, indirect_exchanges_proportions, exchange_maps, agent_maps):
        self.direct_exchanges_proportions = direct_exchanges_proportions
        self.indirect_exchanges_proportions = indirect_exchanges_proportions
        self.exchange_maps = exchange_maps
        self.agent_maps = agent_maps


def main(t_max=2, map_height=10, map_width=10, graphics=False):

    np.random.seed(0)

    # tqdm.tqdm_gui.write("Producing data...")

    eco = model.Model(map_width=map_width, map_height=map_height)

    direct_exchanges_proportions = np.zeros((t_max, 3))
    indirect_exchanges_proportions = np.zeros((t_max, 3))

    idx = np.arange(eco.n, dtype=int)

    agent_maps = None
    exchange_maps = None

    # Place agents and stuff...
    eco.setup()

    if graphics:

        agent_maps = np.zeros((t_max, map_width, map_height))
        exchange_maps = np.zeros((t_max, map_width, map_height, 3))

        # Save initial positions
        agent_maps[0] = eco.agent_map.copy()

    # for t in tqdm.tqdm(range(t_max)):

    for t in range(t_max):

        eco.reset()

        np.random.shuffle(idx)

        for i in idx:

            print("main:", i, type(i), "\n")

            # move agent, then make them proceeding to exchange
            eco.move(i)
            eco.encounter(i)

        # -------------------- #
        # For saving...
        # ------------------- #
        if graphics:

            agent_maps[t] = eco.agent_map.copy()

            exchange_maps[t] = eco.exchange_map.copy()

        # -----------------  #

        # ---------- #
        # Do some stats...

        # for each "t" we compute the proportion of direct choices
        eco.compute_choices_proportions()

        direct_exchanges_proportions[t, :] = eco.direct_choices_proportions.copy()
        indirect_exchanges_proportions[t, :] = eco.indirect_choices_proportions.copy()

    # Finally we compute the direct choices mean for each type
    # of agent and return it as well as the direct choices proportions

    # tqdm.tqdm_gui.write("\nDone!")

    return Result(direct_exchanges_proportions=direct_exchanges_proportions,
                  indirect_exchanges_proportions=indirect_exchanges_proportions,
                  exchange_maps=exchange_maps, agent_maps=agent_maps)


if __name__ == "__main__":

    main()
