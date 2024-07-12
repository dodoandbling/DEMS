import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'normal'
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['figure.figsize'] = (5,4)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.loc'] = 'lower right'
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = True#默认为false，此处设置为TRUE
plt.rcParams['font.family'] = 'Times New Roman'

labels = ['IEEE-14', 'New England-39','IEEE-57','IEEE-118']

lrmf = [0.00071738, 0.0030237,0.0029102,0.0029609]
godec = [0.00064028, 0.00240823,0.00313493,0.005459396]
admm = [0.032516, 0.047796,0.067483,0.1235]



import numpy as np
import matplotlib.pyplot as plt
from pylab import *

 
index1 = np.arange(1,3,1)
index2 = np.arange(2,5,1)
width = 0.2
 
plt.bar(x=index1, height=godec[0:1], width=width, color='yellow', label=u'GoDec')
plt.bar(x=index1, height=lrmf[0:1], width=width,bottom=godec[0:1], color='green', label=u'LMaFit')
plt.bar(x=index1, height=admm[0:1], width=width,bottom=lrmf[0:1], color='red', label=u'ADMM')   

plt.bar(x=index2, height=lrmf[2:3], width=width, color='yellow', label=u'GoDec')
plt.bar(x=index2, height=godec[2:3], width=width,bottom=lrmf[2:3], color='green', label=u'LMaFit')
plt.bar(x=index2, height=admm[2:3], width=width,bottom=godec[2:3], color='red', label=u'ADMM') 


plt.xlabel('Test system')
plt.ylabel('CPU computational time(s)')
# plt.title('2019年销售报告')
plt.legend(loc='best')

plt.show()
