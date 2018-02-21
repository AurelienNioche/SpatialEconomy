from pylab import plt, np
from os import path, mkdir
from analysis.import_data import DataImporter


class GraphProportionChoices(object):
    def __init__(self):

        pass

    @classmethod
    def plot(cls, session_suffix, eco_idx):

        data_importer = DataImporter(session_suffix=session_suffix)
        parameters = data_importer.get_parameters(eco_idx)
        # direct_exchange = data_importer.get_exchanges(eco_idx, "direct")
        indirect_exchange = data_importer.get_exchanges(eco_idx, "indirect")

        t_max = len(indirect_exchange)

        agents_proportions = np.zeros((t_max, 3))

        for i in range(t_max):

            for j in range(3):
                agents_proportions[i, j] = indirect_exchange[i][j]

        cls.draw(agents_proportions=agents_proportions, t_max=t_max, eco_idx=eco_idx, parameters=parameters)

    @classmethod
    def draw(cls, t_max, agents_proportions, eco_idx, parameters):

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
                parameters["x0"],
                parameters["x1"],
                parameters["x2"],
                parameters["movement_area"],
                parameters["vision_area"],
                parameters["alpha"],
                parameters["tau"]
                          ), fontsize=12)

        if not path.exists("../../figures"):
            mkdir("../../figures")

        plt.savefig("../../figures/figure_{}.pdf".format(eco_idx))
        plt.show()


if __name__ == "__main__":

    """
    Examples
    6022: multiple interruptions
    4515: monetary
    """

    idx = 7772

    GraphProportionChoices.plot(
        session_suffix="2016_11_17", eco_idx=34)