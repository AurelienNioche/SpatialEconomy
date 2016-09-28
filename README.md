# SpatialEconomy
What happens with agents that can freely move on a map?

For generating data on a calculation server:

- be sure that you already compiled the cython extension using setup.py

- launch parameters_generator.py
  * generate pickle objects containing list of dictoniary, each dictionary containing paramaters for one economy.
  * generate job scripts for calculation server.

- launch meta_launcher.sh
 * compute on server.
