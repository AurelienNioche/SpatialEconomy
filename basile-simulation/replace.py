import re 

for i in range(10):
	f = open("basile-simulation_{}.sh".format(i), 'r')
	f2 = f.read()
	replaced = re.sub("03:15:00", "06:00:00",f2)
	f.close()	
	f = open("basile-simulation_{}.sh".format(i), 'w')
	f.write(replaced)
