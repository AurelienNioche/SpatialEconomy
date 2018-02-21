from sqlite3 import connect

# ------------------------------------------------||| IMPORT DATA  |||----------------------------------------------- #


class DataImporter(object):

    def __init__(self, session_suffix):

        self.connection = connect("../../data/data_{}.db".format(session_suffix))
        self.cursor = self.connection.cursor()

    def get_exchanges(self, eco_idx, type_of_exchange):

        query = "SELECT `x0`, `x1`, `x2` FROM `exchanges_{}` " \
                "WHERE `exchange_type`='{}' ORDER BY `t` ASC" \
            .format(eco_idx, type_of_exchange)
        self.cursor.execute(query)

        return self.cursor.fetchall()

    def get_eco_idx(self):

        query = "SELECT `eco_idx` FROM `parameters`"
        self.cursor.execute(query)
        return [i[0] for i in self.cursor.fetchall()]

    def get_parameters(self, eco_idx):

        query = "SELECT `vision_area`, `movement_area`, `stride`," \
                "`width`, `height`, `x0`, `x1`, `x2`, `alpha`, `tau`, `t_max` FROM `parameters` WHERE `eco_idx`={}" \
            .format(eco_idx)
        self.cursor.execute(query)

        out = self.cursor.fetchone()

        return {
            "vision_area": out[0],
            "movement_area": out[1],
            "stride": out[2],
            "width": out[3],
            "height": out[4],
            "x0": out[5],
            "x1": out[6],
            "x2": out[7],
            "alpha": out[8],
            "tau": out[9],
            "t_max": out[10]
        }

    def close(self):

        self.connection.close()
