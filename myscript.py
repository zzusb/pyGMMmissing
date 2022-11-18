import sys
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(r"F:/code/python/GMM_EX/GMM_section1/")
from gmm_missing import GMM

if __name__=="__main__":
	f=open(r'F:/code/python/GMM_EX/GMM_section1/new_USC00040161_2000-2018.csv','r')
	ds=pd.read_csv(f,parse_dates=["TIME"],infer_datetime_format=True)
	XNAN=np.array(ds[["PR","TMAX","TMIN"]])
	gmm=GMM(XNAN,5)
	gmm.fit2()
	l_=np.linspace(1,len(gmm.newl),len(gmm.newl))
	plt.plot(l_,gmm.newl,'c*-')
	plt.xlabel('Time')
	plt.ylabel('ln LL')
	plt.title('The variation of ln-likelihood')
	plt.savefig('./ln-likelihood.png',dpi=300)
	plt.clf()

	with open('./GMM.txt','w') as ff:

		ff.write("The mean of guassain mixture is = \n")
		ff.write(str(gmm.newmean_arr))
		ff.write('\n')

		ff.write("The sigma of guassain mixture is = \n")
		ff.write(str(gmm.newsigma_arr))
		ff.write('\n')

		ff.write ("The weight (phi) of guassain mixture is = \n")
		ff.write(str(gmm.newphi))
		ff.write('\n')

		ff.write ("The latent Z variable is = \n")
		ff.write (str(gmm.neww))
		ff.write('\n')
	print ('')

