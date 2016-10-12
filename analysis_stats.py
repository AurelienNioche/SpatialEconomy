import numpy as np
from multiprocessing import Pool
from collections import OrderedDict
from save.save_db_dic import Database
from save.import_data import import_parameters
from analysis_graphics import GraphProportionChoices
from pylab import plt
from os import path, mkdir
import datetime


class Analyst(object):

    def __init__(self, session_suffix, figure_folder):
        self.session_suffix = session_suffix
        self.db = Database(database_name="results_{}".format(session_suffix))
        self.n = len(self.db.read_column(column_name='ID'))
        self.parameters = self.db.get_column_list()

        self.parameters_to_test = ['alpha', 'tau', 'area_move', 'area_vision']

        self.figure_folder = figure_folder

        print("Parameters:", self.parameters)
        print()

    def create_figure_folder(self):

        if not path.exists(self.figure_folder):
            mkdir(self.figure_folder)

    def compute_min_max(self):

        for i in self.parameters_to_test:
            self.get_n_money_state_according_to_parameter(i)

        print("*"*10)
        print("And under the hypothesis that a0 = a1 = a2?")
        print("*" * 10)
        print()
        self.stats_about_economies_with_equal_fd()
        for i in self.parameters_to_test:
            self.get_n_money_state_according_to_parameter_and_equal_fd(i)

    def get_n_money_state_according_to_parameter(self, parameter):

        assert parameter in self.parameters, 'You ask for a parameter that is not in the list...'

        parameter_values = np.unique(self.db.read_column(column_name='{}'.format(parameter)))
        print("Possible values for {}:".format(parameter), parameter_values)
        print("Possible values for {}:".format(parameter), "min", np.min(parameter_values),
              "max", np.max(parameter_values))

        values_with_money = \
            [i[0] for i in self.db.read(query="SELECT `{}` FROM `data` WHERE m_sum > 0".format(parameter))]
        print("Values with money for {}:".format(parameter), "min", np.min(values_with_money),
              "max", np.max(values_with_money))
        print()

    def stats_about_economies_with_equal_fd(self):

        print("n sample:", self.n)
        print("n with equal fundamental structure:",
              len(self.db.read(query="SELECT `ID` FROM `data` WHERE a0 = a1 AND a1 = a2")))
        print("n with equal fundamental structure and more than one moneraty state:",
              len(self.db.read(query="SELECT `ID` FROM `data` WHERE m_sum > 0 AND a0 = a1 AND a1 = a2")))
        print()

    def get_n_money_state_according_to_parameter_and_equal_fd(self, parameter):

        assert parameter in self.parameters, 'You ask for a parameter that is not in the list...'

        parameter_values = np.unique(self.db.read_column(column_name='{}'.format(parameter)))
        print("Possible values for {}:".format(parameter), parameter_values)
        print("Possible values for {}:".format(parameter), "min", np.min(parameter_values),
              "max", np.max(parameter_values))

        values_with_money = \
            [i[0] for i in self.db.read
             (query="SELECT `{}` FROM `data` WHERE m_sum > 0 AND a0 = a1 AND a1 = a2".format(parameter))]
        print("Values with money for {}:".format(parameter), "min", np.min(values_with_money),
              "max", np.max(values_with_money))
        print()

    def represent_var_according_to_parameter(self, var, supplementary_condition="", normalize=True):

        assert var in self.parameters, ""

        t_max = self.db.read_column(column_name='t_max')[0]

        results = {}
        std = {}

        for parameter in self.parameters_to_test:

            parameter_values = np.unique(self.db.read_column(column_name='{}'.format(parameter)))
            print("Possible values for {}:".format(parameter), parameter_values)
            print("Possible values for {}:".format(parameter), "min", np.min(parameter_values),
                  "max", np.max(parameter_values))

            average_m_sum = OrderedDict()
            std_m_sum = OrderedDict()

            for v in parameter_values:

                m_sum = \
                    [i[0] for i in self.db.read
                     (query="SELECT `{}` FROM `data` WHERE {} = {}{}"
                      .format(var, parameter, v, supplementary_condition))]
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

    def select_best_economy(self):

        m_sum = \
            [i[0] for i in self.db.read
             (query="SELECT `m_sum` FROM `data`")]
        max_m_sum = np.max(m_sum)

        idx_max_m_sum = self.db.read(query="SELECT `idx` FROM `data` WHERE `m_sum` = {}".format(max_m_sum))[0][0]
        print("Economy idx with the greatest number of monetary state:", idx_max_m_sum)

        economy_suffix = "{}_idx{}".format(self.session_suffix, idx_max_m_sum)

        print("Parameters", import_parameters(economy_suffix))

        GraphProportionChoices.plot(suffix="{}_idx{}".format(self.session_suffix, idx_max_m_sum))

    def plot_var_against_parameter(self, var, results, std, comment=None):

        for parameter in results.keys():

            x = np.asarray([i for i in results[parameter].keys()])

            y = np.asarray([i for i in results[parameter].values()])
            y_std = np.asarray([i for i in std[parameter].values()])

            # Rename and reorder

            if parameter == 'q_information':
                parameter = "Information quantity"

            elif parameter == "area_vision":
                parameter = 'Vision area'

            elif parameter == "area_move":

                parameter = "Displacement area"

            elif parameter == 'epsilon':
                y = y[::-1]
                y_std = y_std[::-1]
                parameter = 'gamma'

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
        plt.ylim(-0.001, 0.1)

        if not path.exists(fig_folder):
            mkdir(fig_folder)

        plt.savefig("{}/{}".format(fig_folder, fig_name))
        plt.close()


def main(session_suffix):

    a = Analyst(session_suffix, figure_folder="../figures")
    # a.compute_min_max()
    results, std = a.represent_var_according_to_parameter('m_sum')
    a.plot_var_against_parameter('m_sum', results, std)
    # a.represent_var_according_to_parameter('interruptions')
    # a.select_best_economy()

if __name__ == "__main__":

    main(session_suffix="2016-07-29_15-17")
