from pylab import plt, np
from os import path, mkdir

from save.import_data import import_data


class GraphProportionChoices(object):
    def __init__(self):

        pass

    @classmethod
    def plot(cls, suffix, data_folder=None, multi=0):
        if data_folder:
            parameters, direct_exchange, indirect_exchange = import_data(suffix=suffix, data_folder=data_folder)
        else:
            parameters, direct_exchange, indirect_exchange = import_data(suffix=suffix)

        if multi:

            nb_list = len(indirect_exchange)
            t_max = len(indirect_exchange[0])

            agents_proportions = np.zeros((t_max, 3))

            for i in range(nb_list):

                for k in range(t_max):

                    for j in range(3):
                        agents_proportions[k, j] = direct_exchange[i][k][j]

                cls.draw(agents_proportions=agents_proportions, t_max=t_max, suffix=suffix)
        else:

            t_max = len(indirect_exchange)

            agents_proportions = np.zeros((t_max, 3))

            for i in range(t_max):

                for j in range(3):
                    agents_proportions[i, j] = indirect_exchange[i][j]

            cls.draw(agents_proportions=agents_proportions, t_max=t_max, suffix=suffix, parameters=parameters)

    @classmethod
    def draw(cls, t_max, agents_proportions, suffix, parameters):

        color_set = ["green", "blue", "red"]

        for agent_type in range(3):
            plt.plot(np.arange(t_max), agents_proportions[:, agent_type],
                     color=color_set[agent_type], linewidth=2.0, label="Type-{} agents".format(agent_type))

            plt.ylim([-0.1, 1.1])

        plt.xlabel("$t$")
        plt.ylabel("Proportion of indirect exchanges")

        # plt.suptitle('Direct choices proportion per type of agents', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12)

        print(parameters)

        plt.title(
            "Workforce: {}, {}, {};   displacement area: {};   vision area: {};   alpha: {};   tau: {}\n"
            .format(
                parameters["workforce"][0],
                parameters["workforce"][1],
                parameters["workforce"][2],
                parameters["area"],
                parameters["vision"],
                parameters["alpha"],
                parameters["tau"]
                          ), fontsize=12)

        if not path.exists("../figures"):
            mkdir("../figures")

        plt.savefig("../figures/figure_{}.pdf".format(suffix))
        plt.show()


if __name__ == "__main__":

    """
    Examples
    6022: multiple interruptions
    4515: monetary
    """

    idx = 7772

    GraphProportionChoices.plot(
        suffix="2016-07-29_15-17_idx{}".format(idx),
        data_folder="/Users/M-E4-ANIOCHE/Documents/SpatialEconomy-data/avakas-data-2016-07-29_15-17")