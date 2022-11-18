import numpy as np
import csv
from matplotlib import pyplot as plt

np.random.seed(0)
X1 = np.random.multivariate_normal([20, 3,10], [[0.5, 0,2], [0, 0.8,3],[1,0,1]], 300)
X2 = np.random.multivariate_normal([10, 3,6], np.identity(3), 700)
X = np.vstack((X1,X2))
np.random.shuffle(X)

#f1=open(r'F:\code\python\GMM_EX\GMM_section1/3Dsamples/3-D_data.csv','w',newline='')
#csv_writer=csv.writer(f1)
#csv_writer.writerows(X)
#f1.close()

# set X value randomly to zero  
Xnan=X.copy()
ratioNan=0.3
num,dim=X.shape
Xration=np.random.rand(num,dim)
Xnan[Xration<ratioNan]=np.nan
# remove the rows that have only NaN values
#gg=np.isnan(Xnan).sum(axis=1)
#Xnan2=np.delete(Xnan,gg==max(gg),axis=0)
XNAN=np.delete(Xnan,np.isnan(Xnan).sum(axis=1)==max(np.isnan(Xnan).sum(axis=1)),axis=0)

#f=open(r'F:\code\python\GMM_EX\GMM_section1/3Dsamples/3-D_nandata.csv','w',newline='')
#csv_writer=csv.writer(f)
#csv_writer.writerows(XNAN)
#f.close()

fig=plt.figure()
ax1=plt.axes(projection='3d')
ax1.scatter3D(X1[:,0],X1[:,1],X1[:,2],marker='o',c='green',s=30,alpha=0.1)
ax1.scatter3D(X2[:,0],X2[:,1],X2[:,2],marker='o',c='orange',s=30,alpha=0.1)
ax1.scatter(20,3,10,marker='D',c='black',edgecolors='black',s=30)
ax1.scatter(10,3,6,marker='D',c='black',edgecolors='black',s=30)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.tight_layout()
#plt.savefig(r'F:\code\python\GMM_EX\GMM_section1/3Dsamples/3-D_nandata_plot.png',dpi=600)
plt.show()