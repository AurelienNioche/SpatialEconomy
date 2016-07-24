from module.save_array import Database
from time import time
import numpy as np
from tqdm import tqdm


def write(data, table_name='data', database_name='data', descr=''):

    n_columns = data[0].shape[1]

    t0 = time()

    db = Database(database_name)
    db.remove(table_name)
    db.create_table(n_columns=n_columns, table_name=table_name)

    pbar = tqdm(data)
    pbar.set_description("Saving table {}".format(descr))
    for array in pbar:
        db.fill_table(array=array, table_name=table_name)

    t1 = time()

    # print('Time needed for writing', t1 - t0)


def read(n_rows_by_matrix, table_name='data', database_name='data'):

    t0 = time()

    db = Database(database_name=database_name)
    data = db.read(n_rows=n_rows_by_matrix, table_name=table_name)

    t1 = time()

    # print('Time needed for reading', t1 - t0)

    return data


def create_fake_data(array_size, n):

    list_array = []

    for i in range(n):
        data = np.random.random(array_size)
        list_array.append(data)

    return list_array


def main():

    fake_data = create_fake_data(array_size=(60, 60), n=1000)

    write(fake_data)
    obtained_data = read(n_rows_by_matrix=60)
    print("n obtained data", len(obtained_data))

if __name__ == "__main__":

    main()
