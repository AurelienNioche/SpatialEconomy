from analysis.analysis_unique_eco import GraphProportionChoices
from pylab import plt
import datetime
from sqlite3 import connect
from os import path, mkdir
from collections import OrderedDict
import numpy as np


# ------------------------------------------------||| ANALYST  |||----------------------------------------------- #

class Analyst(object):

    def __init__(self, session_suffix, figure_folder):

        self.session_suffix = session_suffix
        self.figure_folder = figure_folder

        self.parameters_to_test = ['alpha', 'tau', 'vision_area', 'movement_area']

        self.connection = connect("../../results/results_{}.db".format(session_suffix))
        self.cursor = self.connection.cursor()

        self.parameters = self.get_parameters()

        print("Parameters:", self.parameters)
        print()

    def get_parameters(self):

        self.cursor.execute("PRAGMA table_info(`results`)")

        return [i[1] for i in self.cursor.fetchall()]

    def get_unique_values_for_parameter(self, parameter):

        assert parameter in self.parameters, 'You ask for a parameter that is not in the list...'

        self.cursor.execute("SELECT `{}` from `results`".format(parameter))
        parameter_values = np.unique(self.cursor.fetchall())

        return parameter_values

    def create_figure_folder(self):

        if not path.exists(self.figure_folder):
            mkdir(self.figure_folder)

    def get_n_money_state_according_to_parameter(self, parameter, uniform_repartition=False):

        if uniform_repartition:
            sup_condition = " AND `x0`=`x1` AND `x1`=`x2`"
        else:
            sup_condition = ""

        parameter_values = self.get_unique_values_for_parameter(parameter)

        print("Possible values for {}:".format(parameter), parameter_values)
        print("Possible values for {}:".format(parameter), "min", np.min(parameter_values),
              "max", np.max(parameter_values))

        self.cursor.execute("SELECT `{}` FROM `results` WHERE m_sum > 0 {}".format(parameter, sup_condition))

        values_with_money = \
            [i[0] for i in self.cursor.fetchall()]

        print("Values with money for {}:".format(parameter), "min", np.min(values_with_money),
              "max", np.max(values_with_money))
        print()

    def stats_about_economies_with_equal_fd(self):

        self.cursor.execute("SELECT `ID` from `results`")
        print("n sample:", len(self.cursor.fetchall()))

        self.cursor.execute("SELECT `ID` FROM `results` WHERE `x0`=`x1` AND `x1`=`x2`")
        print("n with equal fundamental structure:",
              len(self.cursor.fetchall()))

        self.cursor.execute("SELECT `ID` FROM `results` WHERE `m_sum` > 0 AND `x0`=`x1` AND `x1`=`x2`")
        print("n with equal fundamental structure and more than one moneraty state:",
              len(self.cursor.fetchall()))
        print()

    def represent_var_according_to_parameter(self, var, uniform_repartition=False, normalize=True):

        if uniform_repartition:
            sup_condition = " AND `x0`=`x1` AND `x1`=`x2`"
        else:
            sup_condition = ""

        assert var in self.parameters, ""

        self.cursor.execute("SELECT `t_max` from `results`")
        t_max = self.cursor.fetchone()[0]

        results = {}
        std = {}

        for parameter in self.parameters_to_test:

            self.cursor.execute("SELECT `{}` from `results`".format(parameter))

            parameter_values = np.unique(self.cursor.fetchall())
            print("Possible values for {}:".format(parameter), parameter_values)
            print("Possible values for {}:".format(parameter), "min", np.min(parameter_values),
                  "max", np.max(parameter_values))

            average_m_sum = OrderedDict()
            std_m_sum = OrderedDict()

            for v in parameter_values:

                self.cursor.execute("SELECT `{}` FROM `results` WHERE {}={}{}"
                                    .format(var, parameter, v, sup_condition))

                m_sum = \
                    [i[0] for i in self.cursor.fetchall()]
                m_sum = np.asarray(m_sum)
                if normalize:
                    m_sum = m_sum/t_max

                average_m_sum[v] = np.mean(m_sum)
                std_m_sum[v] = np.std(m_sum)

            print("Average '{}' for {}".format(var, parameter), average_m_sum)

            results[parameter] = average_m_sum
            std[parameter] = std_m_sum

            print()

        return results, std

    def select_best_economy(self, session_suffix):

        self.cursor.execute("SELECT `m_sum` FROM `results`")

        m_sum = \
            [i[0] for i in self.cursor.fetchall()]
        max_m_sum = np.max(m_sum)

        self.cursor.execute("SELECT `eco_idx` FROM `results` WHERE `m_sum` = {}".format(max_m_sum))
        idx_max_m_sum = self.cursor.fetchone()[0]
        print("Economy idx with the greatest number of monetary state:", idx_max_m_sum)

        self.cursor.execute("SELECT `alpha`, `tau`, `vision_area`, `movement_area` "
                            "FROM `results` WHERE `m_sum` = {}".format(max_m_sum))
        print("Parameters", self.cursor.fetchone())

        GraphProportionChoices.plot(session_suffix=session_suffix, eco_idx=idx_max_m_sum)

    def plot_var_against_parameter(self, var, results, std, comment=None):

        for parameter in results.keys():

            x = np.asarray([i for i in results[parameter].keys()])

            y = np.asarray([i for i in results[parameter].values()])
            y_std = np.asarray([i for i in std[parameter].values()])

            # Rename and reorder

            # if parameter == 'q_information':
            #     parameter = "Information quantity"
            #
            # elif parameter == "area_vision":
            #     parameter = 'Vision area'
            #
            # elif parameter == "area_move":
            #
            #     parameter = "Displacement area"
            #
            # elif parameter == 'epsilon':
            #     y = y[::-1]
            #     y_std = y_std[::-1]
            #     parameter = 'gamma'

            if var == "m_sum":
                var = "Proportion of monetary states"

            parameter = parameter.capitalize()

            x_label = "{}".format(parameter)

            y_label = "{}".format(var)

            fig_title = "{} according to {}".format(var, parameter)

            if comment:
                fig_name_comment = comment.replace(" ", "_")
                fig_name = "{}_against_{}_{}_{}.pdf".format(var, parameter,fig_name_comment
                                                        , datetime.date.today())
            else:
                fig_name = "{}_against_{}_{}.pdf".format(var, parameter, datetime.date.today())

            self.plot(x=x, y=y, x_label=x_label, y_label=y_label, y_std=y_std,
                      fig_title=fig_title, fig_name=fig_name, fig_folder=self.figure_folder, comment=comment)

    @staticmethod
    def plot(x, y, y_std, x_label, y_label, fig_title, fig_folder, fig_name, comment=None):

        plt.figure(figsize=(10, 10))

        plt.plot(x, y, c='b', lw=2)
        plt.plot(x, y + y_std, c='b', lw=.5)
        plt.plot(x, y - y_std, c='b', lw=.5)
        plt.fill_between(x, y + y_std, y - y_std, color='b', alpha=.1)

        plt.xlabel("\n{}".format(x_label), fontsize=12)
        plt.ylabel("{}\n".format(y_label), fontsize=12)
        if comment:
            plt.title("{}\n({})\n".format(fig_title, comment))
        else:
            plt.title("{}\n".format(fig_title))

        # if comment:
        #
        #     plt.text(x=min(x) + (max(x) - min(x)) * 0.5, y=min(y) + (max(y) - min(y)) * 0.5,
        #              s="{}".format(comment))

        plt.xlim(min(x), max(x))
        plt.ylim(-0.001, 1.0)

        if not path.exists(fig_folder):
            mkdir(fig_folder)

        plt.savefig("{}/{}".format(fig_folder, fig_name))
        plt.close()


def main(session_suffix):

    a = Analyst(session_suffix, figure_folder="../../figures")
    # a.compute_min_max()
    results, std = a.represent_var_according_to_parameter('m_sum')
    a.plot_var_against_parameter('m_sum', results, std)
    results, std = a.represent_var_according_to_parameter('m_sum', uniform_repartition=True)
    a.plot_var_against_parameter('m_sum', results=results, std=std, comment='uniform_repartition')
    # a.represent_var_according_to_parameter('interruptions')
    # a.select_best_economy()

if __name__ == "__main__":

    main(session_suffix="2016_11_17")
