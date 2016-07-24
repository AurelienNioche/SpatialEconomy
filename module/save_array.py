from sqlite3 import connect
import numpy as np
from os import path
from time import time


class Database(object):

    def __init__(self, database_name="db"):

        self.db_path = "../data/{}.db".format(database_name)

        self.connexion = connect(self.db_path)
        self.cursor = self.connexion.cursor()

        self.fill_queries = dict()

    def create_table(self, n_columns, table_name='data'):

        query = "CREATE TABLE `{}` (" \
                "ID INTEGER PRIMARY KEY AUTOINCREMENT, ".format(table_name)

        fill_query = "INSERT INTO '{}' (".format(table_name)
        for i in range(n_columns):

            query += "`{}` REAL, ".format(i)
            fill_query += "`{}`, ".format(i)

        query = query[:-2]
        fill_query = fill_query[:-2]
        query += ")"
        fill_query += ") VALUES ("

        for i in range(n_columns):
            fill_query += "?, "
        fill_query = fill_query[:-2]
        fill_query += ")"

        self.fill_queries[table_name] = fill_query

        self.cursor.execute(query)
        self.connexion.commit()

    def fill_table(self, array, table_name='data'):

        self.cursor.executemany(self.fill_queries[table_name], array)

    def remove(self, table_name='data'):

        if self.has_table(table_name):

            q = "DROP TABLE `{}`".format(table_name)
            self.cursor.execute(q)
            self.connexion.commit()

    def has_table(self, table_name):

        table_exists = 0

        if path.exists(self.db_path):

            # noinspection SqlResolve
            already_existing = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

            if already_existing:

                already_existing = [i[0] for i in already_existing]

                if table_name in already_existing:
                    table_exists = 1

        else:
            pass

        return table_exists

    def read(self, n_rows, table_name='data'):
        
              
        raw_data = self.cursor.execute("SELECT * from {}".format(table_name)).fetchall()
        n_columns = len(raw_data[0]) - 1  # Do not count ID column
        list_array = []

        cursor = 0
        while cursor < len(raw_data):

            a = np.zeros((n_rows, n_columns))

            for i, element in enumerate(raw_data[cursor:cursor+n_rows]):

                a[i, :] = element[1:]
            list_array.append(a)
            cursor += n_rows

        return list_array

    def __del__(self):

        self.connexion.commit()
        self.connexion.close()
