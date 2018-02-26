from pylab import plt, np

import analysis


def run(data):
    draw(indirect_exchanges_proportions=data.indirect_exchanges_proportions, parameters=data.parameters,
         file_name=data.file_name)


def draw(indirect_exchanges_proportions, parameters, file_name):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    color_set = ["C0", "C1", "C2"]

    for agent_type in range(3):
        ax.plot(np.arange(parameters.t_max), indirect_exchanges_proportions[:, agent_type],
                 color=color_set[agent_type], linewidth=2.0, label="Type-{} agents".format(agent_type))

    ax.set_ylim((-0.1, 1.1))

    ax.set_xlabel("$t$")
    ax.set_ylabel("Proportion of indirect exchanges")

    # plt.suptitle('Direct choices proportion per type of agents', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)

    ax.set_title(
        "\nWorkforce: {}, {}, {}; movement area: {} stride: {};\nvision area: {};   alpha: {};   tau: {}\n"
        .format(
            parameters.x0,
            parameters.x1,
            parameters.x2,
            parameters.movement_area,
            parameters.stride,
            parameters.vision_area,
            parameters.alpha,
            parameters.tau
        ), fontsize=12)

    if file_name:
        plt.text(0.005, 0.005, file_name, transform=fig.transFigure, fontsize='x-small', color='0.5')

        plt.savefig("{}/separate_indirect_exchanges_proportions_{}.pdf"
                    .format(analysis.parameters.fig_folder, file_name))

    plt.show()
