from pylab import np, plt
import enum
import os

import analysis


class MoneyAnalysis:

    def __init__(self, m0, m1, m2, interruptions):

        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.interruptions = interruptions


class MoneyAnalyst(object):

    money_threshold = .75

    @classmethod
    def _test_for_money_state(cls, direct_exchange, indirect_exchange):

        money = -1

        # Money = 0?
        # type '0' should use direct exchange
        cond0 = direct_exchange[0] > cls.money_threshold

        # type '1' should use indirect exchange
        cond1 = indirect_exchange[1] > cls.money_threshold

        # type '2' should use direct exchange
        cond2 = direct_exchange[2] > cls.money_threshold

        if (cond0 * cond1 * cond2) == 1:

            money = 0

        else:

            # Money = 1?
            cond0 = direct_exchange[0] > cls.money_threshold
            cond1 = direct_exchange[1] > cls.money_threshold
            cond2 = indirect_exchange[2] > cls.money_threshold

            if (cond0 * cond1 * cond2) == 1:

                money = 1

            else:

                # Money = 2?
                cond0 = indirect_exchange[0] > cls.money_threshold
                cond1 = direct_exchange[1] > cls.money_threshold
                cond2 = direct_exchange[2] > cls.money_threshold

                if (cond0 * cond1 * cond2) == 1:
                    money = 2

        return money

    @classmethod
    def run(cls, direct_exchange, indirect_exchange, t_max):

        money_time_line = np.zeros(t_max)
        money = {0: 0, 1: 0, 2: 0, -1: 0}
        interruptions = 0

        for t in range(t_max):

            money_t = cls._test_for_money_state(
                direct_exchange=direct_exchange[t],
                indirect_exchange=indirect_exchange[t])
            money_time_line[t] = money_t
            money[money_t] += 1

            if t > 0:

                cond0 = money_t == -1
                cond1 = money_time_line[t-1] != -1
                interruptions += cond0 * cond1

        return MoneyAnalysis(
            m0=money[0],
            m1=money[1],
            m2=money[2],
            interruptions=interruptions)


def separate_plots_for_indirect_exchanges(results_pool):

    data = results_pool.data

    for i, d in enumerate(data):

        param_str = {
            "x": d.parameters.x0,  # Suppose equal repartition
            "stride": d.parameters.stride,
            "movement_area": d.parameters.movement_area,
            "vision_area": d.parameters.vision_area,
            "alpha": float("{:.2f}".format(d.parameters.alpha)),
            "tau": float("{:.2f}".format(d.parameters.tau)),
            "map_width": d.parameters.map_width,
            "map_height": d.parameters.map_height
        }

        fig = plt.figure()
        ax = fig.add_subplot(111)
        lines = ax.plot(d.indirect_exchanges_proportions)
        ax.set_ylim(0., 1.)
        ax.set_xlabel("t")
        ax.set_ylabel("Indirect exchanges proportion")
        plt.text(0.005, 0.005, param_str, transform=fig.transFigure, fontsize='x-small', color='0.5')
        plt.legend(lines, ("type-0", "type-1", "type-2"))
        plt.title("Eco {} of pool {}".format(i, results_pool.file_name))

        plt.savefig(
            "{}/{}/ind/{}_indirect_{}.pdf"
            .format(analysis.parameters.fig_folder, results_pool.file_name, results_pool.file_name, i))
        plt.show()


def summary_plots(results_pool):

    class X(enum.Enum):
        alpha = enum.auto()
        tau = enum.auto()
        vision_area = enum.auto()
        x = enum.auto()

    parameters = results_pool.parameters
    data = results_pool.data

    x = {
        X.alpha: [],
        X.tau: [],
        X.vision_area: [],
        X.x: []
    }
    y = []

    for d in data:
        a = MoneyAnalyst.run(
            t_max=parameters.t_max,
            direct_exchange=d.direct_exchanges_proportions,
            indirect_exchange=d.indirect_exchanges_proportions
        )

        x[X.vision_area].append(d.parameters.vision_area)
        x[X.tau].append(d.parameters.tau)
        x[X.alpha].append(d.parameters.alpha)
        x[X.x].append(d.parameters.x0 + d.parameters.x1 + d.parameters.x2)

        y.append(
            a.m0 + a.m1 + a.m2
        )

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(221)
    ax.scatter(x[X.tau], y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"$\tau$")

    ax = fig.add_subplot(222)
    ax.scatter(x[X.alpha], y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"$\alpha$")

    ax = fig.add_subplot(223)
    ax.scatter(x[X.vision_area], y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"vision area")

    ax = fig.add_subplot(224)
    ax.scatter(x[X.x], y, c="black", alpha=0.4, s=15)
    ax.set_ylabel("n monetary states")
    ax.set_xlabel(r"n agents")

    plt.text(0.005, 0.005, results_pool.file_name, transform=fig.transFigure,
             fontsize='x-small', color='0.5')

    plt.tight_layout()

    plt.savefig(
        "{}/{}/{}_summary.pdf".format(
            analysis.parameters.fig_folder,
            results_pool.file_name,
            results_pool.file_name))

    plt.show()


def run(results_pool):

    os.makedirs("{}/{}/ind".format(
        analysis.parameters.fig_folder,
        results_pool.file_name), exist_ok=True)

    separate_plots_for_indirect_exchanges(results_pool)
    summary_plots(results_pool)
