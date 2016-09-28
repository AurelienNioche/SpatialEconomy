import subprocess
import json
import sys
from os import listdir
from os.path import isfile, join


class AvakasLauncher(object):

    def __init__(self):

        self.folder = {
            "script": "../avakas_scripts",
            "job_names": ".."
        }

    def load_scripts(self):

        mypath = self.folder["script"]
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        script_names = [f for f in onlyfiles if f[-3:] == ".sh"]

        return script_names

    def save_job_names(self, job_names):

        with open('{}/job_names.json'.format(self.folder["job_names"]), 'w') as f:

            json.dump(job_names, f, indent=4)

    def launch_jobs(self, script_names):

        job_names = []

        for script_name in script_names:

            print("Launch script '{}'...".format(script_name))

            output = subprocess.check_output("qsub {}/{}".format(self.folder["script"], script_name).split())

            print("System answers '{}'.".format(str(output)[:-1]))  # [:-1] is for removing the \n at the end

            print()

            job_names.append(str(output).split(".")[0])  # Keep just the number of the job as ID

        return job_names

    def run(self):

        script_names = self.load_scripts()
        assert len(script_names) > 0, "Can't find any script to launch."
        job_names = self.launch_jobs(script_names)
        self.save_job_names(job_names)


def main():

    if sys.version_info[0] != 2:
        raise Exception("Should use Python 2 for launching jobs on Avakas. \n"
                        "Maybe what you want is using avakas-launcher.SH")

    avakas_launcher = AvakasLauncher()
    avakas_launcher.run()

if __name__ == "__main__":

    main()
