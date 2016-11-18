import apsw
from sqlite3 import connect
from datetime import date
from os import listdir, remove
import numpy as np
from tqdm import tqdm


class DatabaseManager(object):

    @classmethod
    def run(cls):

        data_folder = "/Users/M-E4-ANIOCHE/Desktop/SpatialEconomy-data"
        new_db_name = "data_{}".format(str(date.today()).replace("-", "_"))
        new_db_folder = "/Users/M-E4-ANIOCHE/Desktop"
        db_path = "{}/{}.db".format(new_db_folder, new_db_name)

        # remove("{}/{}.db".format(new_db_folder, new_db_name))

        connection = connect(db_path)
        cursor = connection.cursor()

        # Get the list of the databases containing data
        list_db_name = [i for i in listdir(data_folder) if i[-3:] == ".db"]

        # Create tables

        for i in tqdm(list_db_name):

            old_db_path = "{}/{}".format(data_folder, i)

            query = \
                "ATTACH DATABASE '{}' As old;".format(old_db_path)
            cursor.execute(query)

            # cursor.execute("BEGIN")

            query = "SELECT name FROM old.sqlite_master WHERE type='table'"
            cursor.execute(query)

            table_names = [i[0] for i in cursor.fetchall()]

            if "sqlite_sequence" in table_names:
                table_names.remove("sqlite_sequence")

            for j in table_names:

                if j.startswith("parameters"):

                    # Create parameters table in new db
                    query = \
                        "CREATE TABLE IF NOT EXISTS `parameters` (`ID` INTEGER PRIMARY KEY, " \
                        "`eco_idx` INTEGER, `vision_area` INTEGER, " \
                        "`movement_area` INTEGER, `stride` INTEGER, " \
                        "`width` INTEGER, `height` INTEGER , " \
                        "`x0` INTEGER, `x1` INTEGER, `x2` INTEGER, " \
                        "`alpha` REAL, `tau` REAL, `t_max` INTEGER);"
                    cursor.execute(query)

                else:

                    query = "CREATE TABLE IF NOT EXISTS `exchanges_{}` (ID INTEGER PRIMARY KEY , " \
                            "`exchange_type` TEXT, " \
                            "`t` INTEGER, `x0` REAL, `x1` REAL, `x2` REAL);"\
                        .format(j.split("_")[2])
                    cursor.execute(query)

            # cursor.execute("COMMIT")
            query = \
                "DETACH old"
            cursor.execute(query)

        # Fill tables
        idx_parameter = 0

        for i in tqdm(list_db_name):

            old_db_path = "{}/{}".format(data_folder, i)

            query = \
                "ATTACH DATABASE '{}' As old;".format(old_db_path)
            cursor.execute(query)

            query = "SELECT name FROM old.sqlite_master WHERE type='table'"
            cursor.execute(query)

            table_names = [i[0] for i in cursor.fetchall()]

            if "sqlite_sequence" in table_names:
                table_names.remove("sqlite_sequence")

            for j in table_names:

                query = "SELECT * FROM old.'{}';".format(j)
                cursor.execute(query)
                dat = cursor.fetchall()

                if j.startswith("parameters"):

                    eco_idx = int(j.split("parameters_")[1])

                    for d in dat:
                        query = \
                            "INSERT INTO `parameters` (`ID`, `eco_idx`, `vision_area`, " \
                            "`movement_area`, `stride`, `width`, " \
                            "`height`, `x0`, `x1`, `x2`, " \
                            "`alpha`, `tau`, `t_max`" \
                            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"

                        cursor.execute(query, (idx_parameter, eco_idx) + d[1:])

                        idx_parameter += 1
                else:

                    eco_idx = j.split("_")[2]
                    exchange_type = j.split("_")[0]

                    query = "SELECT MAX(id) FROM exchanges_{}".format(eco_idx)

                    cursor.execute(query)

                    output = cursor.fetchone()
                    if output[0] is None:
                        idx = 0
                    else:
                        idx = output[0] + 1

                    for k, d in enumerate(dat):

                        query = "INSERT INTO `exchanges_{}` (`ID`, `exchange_type`, `t`, `x0`, `x1`, `x2`) " \
                                "VALUES (?, ?, ?, ?, ?, ?);".format(eco_idx)

                        cursor.execute(query, (idx+k, exchange_type) + d)

            query = \
                "DETACH old"
            cursor.execute(query)

        connection.commit()
        connection.close()


def main():
    d = DatabaseManager()
    d.run()


if __name__ == "__main__":
    main()