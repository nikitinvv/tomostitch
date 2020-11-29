import numpy as np
import matplotlib.pyplot as plt
import sys
#/mxn/visitors/vviknik/MDA_Utilities/src/mda2ascii
filename = sys.argv[1]
filenameout = sys.argv[2]

fid = open(filename,'r')
k = 0
plt.figure(figsize=(16,6))
arr = np.zeros(125050,dtype='float32')
for line in fid:
    # print(line)
    if(k==18):
        plt.title(line)
    a = line.strip().split()
    # print(k)

    if(len(a)==8 and k>18):
        arr[k-19]=np.float32(np.array(a))[2]
        # print(k,np.float32(np.array(a))[1])
        # exit()
    k+=1
x=np.arange(0,60000)/1000
plt.plot(x,arr[:60000])
plt.xlabel('us')
plt.xticks(x[::2000])
plt.grid()
plt.ylim([-0.002,0.04])
plt.savefig(filenameout)