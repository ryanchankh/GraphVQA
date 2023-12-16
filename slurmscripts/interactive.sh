#!/ bin / bash −l
#SBATCH −−time=08:00:00    # S p e cif y your r e q u e s t computing time [DD:HH:MM]
#SBATCH −−nodes=1           # S p e cif y number of computing nodes to r e q u e s t
#SBATCH −−ntasks−per−node=4 # Number of cpu c o r e s to r e q u e s t .
#SBATCH −−mem=32g
#SBATCH −−gres=gpu:a5000:1