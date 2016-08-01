import numpy as np
from multiprocessing import Pool
from save.save_db_dic import Database


class Analyst(object):

    def __init__(self, session_suffix):

        self.db = Database(database_name="results_{}".format(session_suffix))
        self.n = len(self.db.read_column(column_name='ID'))
        self.parameters = self.db.get_column_list()

        print("Parameters:", self.parameters)
        print()

    def compute_min_max(self):

        parameters_to_test = ['alpha', 'tau', 'area_move', 'area_vision']
        for i in parameters_to_test:
            self.get_n_money_state_according_to_parameter(i)

        print("*"*10)
        print("And under the hypothesis that a0 = a1 = a2?")
        print("*" * 10)
        print()
        self.stats_about_economies_with_equal_fd()
        for i in parameters_to_test:
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
            [i[0] for i in self.db.read(query="SELECT `{}` FROM `data` WHERE m_sum > 0 AND a0 = a1 AND a1 = a2".format(parameter))]
        print("Values with money for {}:".format(parameter), "min", np.min(values_with_money),
              "max", np.max(values_with_money))
        print()





def main(session_suffix):

    a = Analyst(session_suffix)
    a.compute_min_max()

if __name__ == "__main__":

    main(session_suffix="2016-07-29_15-17")
