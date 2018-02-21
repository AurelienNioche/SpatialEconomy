from subprocess import check_output, CalledProcessError
import argparse

print("Hey!")

parser = argparse.ArgumentParser()
parser.add_argument('begin', type=int,
                    help='A beginning number is required!')
parser.add_argument('end', type=int,
                    help='A end number is required!')

args = parser.parse_args()
begin = args.begin
end = args.end

for i in range(begin, end):

    command = "qdel {}".format(i)
    print("Ask system '{}'".format(command))

    try:
        output = check_output(command.split())
        output = str(output)[:-2]  # [:-2] is for removing the \n at the end
        print("System answers '{}'".format(output))

    except CalledProcessError as e:
        print(e)