from subprocess import check_output, CalledProcessError
import json


class JobKiller(object):

    def __init__(self):

        self.folder = {
            "job_names": ".."
        }

    def load_job_names(self):

        with open("{}/job_names.json".format(self.folder["job_names"]), "r") as f:
            job_names = json.load(f)

        return job_names

    @classmethod
    def kill_jobs(cls, job_names):

        for i in job_names:

            command = "qdel {}".format(i)
            print("Ask system '{}'".format(command))

            try:
                output = check_output(command.split())
                output = str(output)[:-2]  # [:-2] is for removing the \n at the end
                print("System answers '{}'".format(output))

            except CalledProcessError as e:
                print(e)

    def run(self):

        job_names = self.load_job_names()
        self.kill_jobs(job_names)


def main():

    j = JobKiller()
    j.run()

if __name__ == "__main__":

    main()
