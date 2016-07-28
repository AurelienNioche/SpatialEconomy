import re


def modify_sh(prefix, n_files, old, new):

    for i in range(n_files):

        f = open("{}{}.sh".format(prefix, i), 'r')
        content = f.read()
        f.close()

        replaced = re.sub("{}".format(old), "{}".format(new), content)

        f = open("{}{}.sh".format(prefix, i), 'w')
        f.write(replaced)
        f.close()

if __name__ == "__main__":

    modify_sh(prefix="basile-simulation_", n_files=10, old="03:15:00", new="06:00:00")
