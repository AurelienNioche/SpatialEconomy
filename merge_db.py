from os import path, listdir, remove
from tqdm import tqdm
from datetime import date
from save.database import Database
from arborescence.arborescence import Folders


def merge_db(db_folder, new_db_name, db_to_merge):

    assert path.exists(db_folder), '`{}` is a wrong path to db folder, please correct it.'.format(db_folder)

    new_db = Database(folder=db_folder, database_name=new_db_name)

    for db_name in tqdm(db_to_merge):

        db = Database(folder=db_folder, database_name=db_name)
        for table_name in db.get_tables_names():

            columns = db.get_columns(table_name)
            content = db.read_n_rows(table_name=table_name, columns=columns)
            if new_db.has_table(table_name=table_name):
                new_db.remove_table(table_name=table_name)
            new_db.create_table(table_name=table_name, columns=columns)
            new_db.write_n_rows(table_name=table_name, columns=columns, array_like=content)

        db.close()
        remove("{}/{}.db".format(db_folder, db_name))
    
    new_db.close()


def merge_all_db_from_same_folder(db_folder, new_db_name):

    # Be sure that the path of the folder containing the databases is correct.
    assert path.exists(db_folder), 'Wrong path to db folder, please correct it.'

    # Get the list of all the databases
    list_db_name = [i[:-3] for i in listdir(db_folder) if i[-3:] == ".db" and i[:-3] != new_db_name]
    assert len(list_db_name), 'Could not find any db...'

    merge_db(db_folder=db_folder, new_db_name=new_db_name, db_to_merge=list_db_name)


def example_of_merging_db_from_list_of_db():

    new_db = "new_db"
    data_folder = "../data"
    db_to_merge = ["2016_09_29_idx17", "2016_09_29_idx16"]
    merge_db(db_folder=data_folder, new_db_name=new_db, db_to_merge=db_to_merge)


def main():

    db_folder = Folders.folders["data"]
    # new_db_name = "data_{}".format(str(date.today()).replace("-", "_"))
    new_db_name = "data_2016_10_12"

    merge_all_db_from_same_folder(db_folder=db_folder, new_db_name=new_db_name)


if __name__ == "__main__":

    main()