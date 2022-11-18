import sys
sys.path.append(r"F:/code/python/GMM_EX/GMM_section1/")
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

def plotexample(example_data):
	columns=[c for c in example_data.columns.values]
	for i in range (len(columns)-1):
		X=example_data[columns[i]]
		for j in range (i+1,len(columns)-1):
			Y=example_data[columns[j]]
			g=sns.jointplot(x=example_data[columns[i]],y=example_data[columns[j]],data=example_data,shade=True,alpha=1,kind='kde',cmap="Spectral_r",color="#825FB1",shade_lowest=False)
			ax1=g.figure.axes[0]
			ax1.grid(linestyle = '--',color="#D6DBE1")
			sns.scatterplot(x=example_data[columns[i]], y=example_data[columns[j]],hue=example_data[columns[-1]],style=example_data[columns[-1]],palette="Set2",ax=ax1,marker=".",s=30,alpha=0.8)
			plt.xlabel(columns[i])
			plt.ylabel(columns[j])
			plt.tight_layout()
			plt.savefig(r'./out/4d_3gmm/0.5/imputed_%s_%s_with0.5_k_3.png'%(columns[i],columns[j]),dpi=600)
			#plt.show()

if __name__=="__main__":
	zn_path=r"F:/code/python/GMM_EX_copy/GMM_section1/out/4d_3gmm/0.5/K=3/Znk.csv"
	znds1=(pd.read_csv(zn_path,header=None))
	phi=znds1.apply(lambda x:x.sum())
	b=np.array(sorted(enumerate(phi), key=lambda x:x[1]))
	new_phi=[]
	for i in phi:
		new_phi.append(np.where(b[:,1]==i)[0][0])

	znds=pd.read_csv(zn_path,header=None,names=new_phi)
	#print (znds)

	#print (znds)
	zn=np.zeros(len(znds))
	for i in range(len(znds)):
		#print (znds.iloc[i,:])
		zn[i]=znds.iloc[i,:].idxmax(0)
	imputedX_path=r"F:/code/python/GMM_EX_copy/GMM_section1/out/4d_3gmm/0.5/K=3/sampling_AVERAGED.csv"
	imputedXds=pd.read_csv(imputedX_path,names=["x1","x2","x3","x4"])
	zn=zn.astype(int)
	imputedXds["class"]=zn
	#print (imputedXds)

	plotexample(imputedXds)

	print ("")
