import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import subprocess

## From here: https://github.com/kzuiderveld/deeplearning1/blob/master/Improving%20training%20speeds%20using%20Keras%202.ipynb

sb.set_style("darkgrid")
obj = subprocess.Popen("timeout 30 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -lms 500 | sed s/%//g > ./GPU-stats.log",shell=True)
obj.communicate()
gpu = pd.read_csv("./GPU-stats.log") 
gpu.plot()
plt.show()
