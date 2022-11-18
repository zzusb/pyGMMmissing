#-*-coding=utf-8 -*-
import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
import csv
'''
This script is for imputating missing values by GMM-EM algorithm.
Author: XT L

Reference:
[1] Liu et al.,2018. Wind power prediction with missing data using Gaussian process regression and multiple imputation.
'''
class GMM(object):
	def __init__(self,X,k=2,inimethod='kmeans'):
		# dimension
		X = np.asarray(X)
		self.m, self.n = X.shape# row(data size), column(variables)
        # number of mixtures
		self.k = k
		self.inimethod=inimethod 
		# divided X into Xo and Xm
		self.data = X.copy()
		# build a mask matrix
		self.Xo = X[np.isnan(X).sum(axis=1)==min(np.isnan(X).sum(axis=1))]
		self.Xm = X[np.isnan(X).sum(axis=1)!=min(np.isnan(X).sum(axis=1))]
		self.m_o=self.Xo.shape[0]
		self.m_m=self.Xm.shape[0]
		assert self.m_o+self.m_m ==self.m
		print ('The data have been passed successfully')
	def _initparams(self):
		if self.inimethod=='random':
			print ('initial parameters by Random')
			# init mixture means/sigmas
			self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
			#sigma should be a definite matrix
			self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
			for si in range(self.k):
				A = np.random.rand(self.n,self.n)
				B = np.dot(A,A.transpose())
				C = B+B.T # makesure symmetric
				## test whether C is definite                            
				#D = np.linalg.cholesky(C)
				self.sigma_arr[si]=C                                 
			
			self.prec_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
			self.phi = np.ones(self.k)/self.k# The weight of K guassians 
			self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))#z_nk
			#print(self.mean_arr)
			#print(self.sigma_arr）
		elif self.inimethod=='kmeans':
			'''
			The use of K-mean is easy to get into the local optimal values
			'''
			print ('initial parameters by K-means')
			# initial the parameters by the observed Xo
			kmeans= KMeans(n_clusters=self.k,random_state=None).fit(self.Xo)# random_state is to control the centroid initialization in KMeans
			self.mean_arr=np.asmatrix(kmeans.cluster_centers_)
			self.phi=np.ones(self.k)/self.k
			self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
			self.prec_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
			for i in np.unique(kmeans.labels_):
				self.phi[i]=sum((kmeans.labels_== i))/len(kmeans.labels_)
				self.sigma_arr[i]=np.cov(self.Xo[kmeans.labels_== i].T)
			self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))#z_nk

	def fit(self, tol=1e-2):
		self.l=[]
		self._initparams()
		num_iters=0
		ll=1
		previous_ll=0
		previous_ll,previous_den=self.loglikelihood()
		self._fit()#E-step and M-step
		num_iters += 1
		ll,d=self.loglikelihood()
		self.l.append(ll)
		print('Iteration %d: log-likelihood is %.6f'%(num_iters, previous_ll))
		while True:
			self.newl=self.l
			self.newmean_arr=self.mean_arr
			self.newphi=self.phi
			self.newsigma_arr=self.sigma_arr
			self.neww=self.w
			self.news=self.s
			self._fit()#E-step and M-step
			num_iters += 1
			ll,d=self.loglikelihood()
			print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
			if ((ll-previous_ll>tol or ll>previous_ll or ll-previous_ll>0 ) and ll<0):
				self.l.append(ll)
				previous_ll=ll
			else:break
		print('The iteration is finished!')
		#while (ll-previous_ll>tol):
		#	previous_ll,previous_den=self.loglikelihood()#
		#	self._fit()#E-step and M-step
		#	num_iters += 1
		#	ll,d=self.loglikelihood()
		#	self.l.append(ll)
		#	print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
		#print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
		#
	def loglikelihood(self):
		'''
		This function is different from the function in gmm.py
		for every Xo, 
			for j
				calculate the pdf  
		for Xm with missing data,
			for j
				calculate the marginal distribution of the observed data
		'''
		ll = 0
		den=np.ones((self.m,self.k),dtype=float)
		for i in range(len(self.data)):
			if np.isnan(self.data[i,:]).any()==False:# Xo,there is no missing in the record.
				tmpo = 0
				for jo in range(self.k):
					#print(self.sigma_arr[j])
					pdfo=sp.stats.multivariate_normal.pdf(self.data[i,:], 
																self.mean_arr[jo, :].A1, 
															self.sigma_arr[jo, :],allow_singular=True) * self.phi[jo]
					#a=sp.stats.multivariate_normal.pdf([-0.75784,5.65468],[0,3],[[0.5,0],[0,0.8]])
					#b=sp.stats.multivariate_normal.pdf([-0.75784,5.65468],[1,6],[[1,0],[0,1]])
					tmpo += pdfo
					den[i,jo]=pdfo
				ll += np.log(tmpo)
			else:
				tmpm = 0
				#get the observed components
				xm=self.data[i,:]
				xm_o=xm[~np.isnan(xm)] # The observed part of the record with missing value 
				for jm in range (self.k):
					mean_m=self.mean_arr[jm, :].A1
					sigma_m=self.sigma_arr[jm, :]
					pdfm=sp.stats.multivariate_normal.pdf(xm_o, mean_m[~np.isnan(xm)], sigma_m[~np.isnan(xm)][:,~np.isnan(xm)],allow_singular=True) * self.phi[jm]
					tmpm += pdfm
					den[i,jm]=pdfm
				ll += np.log(tmpm)
		## For Xo
		#for io in range(self.m_o):
		#	tmpo = 0
		#	for jo in range(self.k):
		#		#print(self.sigma_arr[j])
		#		pdfo=sp.stats.multivariate_normal.pdf(self.Xo[io, :], 
		#													self.mean_arr[jo, :].A1, 
		#												self.sigma_arr[jo, :]) * self.phi[jo]
		#		tmpo += pdfo
		#		den[io,jo]=pdfo
		#	ll += np.log(tmpo)

		## For Xm
		#for im in range(self.m_m):
		#	tmpm = 0
		#	#get the observed components
		#	xm=self.Xm[im,:]
		#	xm_o=xm[~np.isnan(xm)]
		#	for jm in range (self.k):
		#		mean_m=self.mean_arr[jm, :].A1
		#		sigma_m=self.sigma_arr[jm, :]
		#		pdfm=sp.stats.multivariate_normal.pdf(xm_o, mean_m[~np.isnan(xm)], sigma_m[~np.isnan(xm),~np.isnan(xm)]) * self.phi[jm]
		#		tmpm += pdfm
		#		den[io+im+1,jm]=pdfm
		#	ll += np.log(tmpm)
		return ll,den#返回似然函数和当前数据的概率密度
	def _fit(self):
		self.e_step()
		self.m_step()

	def e_step(self):
		'''
		This function is used to calculate the expectation
		'''
		# 1. update the z_nk
		# because the existing missing values, the pdf can not directly be estimated from sp.stats.multivariate_normal.pdf()
		# calculate w_j^{(i)}, eq（10）of paper [1]
		# according the density aquaired by Xo 
		pre_ll,pre_den=self.loglikelihood()
		for i in range(self.m):
			den=0
			for j in range(self.k):
				#num = sp.stats.multivariate_normal.pdf(self.data[i, :], 
				#										  self.mean_arr[j].A1, 
				#										  self.sigma_arr[j]) * self.phi[j]
				#num=pre_den[i,j]* self.phi[j]
				num=pre_den[i,j]
				den += num
				self.w[i, j] = num
			self.w[i, :] /= den
			assert self.w[i, :].sum() - 1 < 1e-4
		# 2. 
		# calculate the precision matrices
		for jj in range (self.k):
			self.prec_arr[jj]=np.linalg.inv(self.sigma_arr[jj])
		# calculate xm, build a sparse matrix containing the missing xm
		#nummissing=sum(sum(np.isnan(self.Xm)))# The total number of missing 
		#缺失值的index
		a=np.isnan(self.data)
		Index_mi,Index_mj=np.where(a==True)
		self.index_mi=Index_mi
		self.index_mj=Index_mj
		self.s=np.zeros((self.m,self.n,self.k))# missing value
		
		pre_n=-1
		for n in range(len(Index_mi)):
			if pre_n!=Index_mi[n]:
				pre_n=Index_mi[n]
				#for nn in [Index_mj[n]]:
				for jjj in range(self.k):
					missingpart=np.isnan(self.data[Index_mi[n]])
					a_=self.prec_arr[jjj]
					a__=self.prec_arr[jjj,missingpart]
					a___=self.prec_arr[jjj,missingpart][:,missingpart]

					mat=np.linalg.inv(np.asmatrix(self.prec_arr[jjj,missingpart][:,missingpart]))*(self.prec_arr[jjj,missingpart][:,~missingpart])
					#s is the estimation of Xm
					self.s[Index_mi[n],missingpart,jjj]= np.squeeze( self.mean_arr[jjj,missingpart].T-mat*(self.data[Index_mi[n],~missingpart]-self.mean_arr[jjj,~missingpart]).T  )

		# 3. do not calculate the second factor == eq.(12)
		# instead calculate it as required in the M-step
		# Here, the precision matrices based the crrent parameters are precalculate==self.prec_arr
		# Expectation=self.w,self.s,self.prec_arr
		print ('The e-step finished!')
	def m_step(self):
		# replace the missing parts in x by their expectation
		# recompute new parameters: miu, sigma, phi, eq(17),(18),(19) of paper [1]
		# calculate sigma_nk and adding it to sigma 
		for j in range(self.k): 
			# replace the missing values with self.s of E-step
			X_m_step=self.data.copy()
			#pre_n=-1
			for n in range (len(self.index_mi)):
				#if pre_n!=self.index_mi[n]:
				#	pre_n=self.index_mi[n]
					#for nn in [self.index_mj[n]]:
				assert np.isnan(X_m_step[self.index_mi[n],self.index_mj[n]])
				X_m_step[self.index_mi[n],self.index_mj[n]]=self.s[self.index_mi[n],self.index_mj[n],j]
			#self.data=X_m_step
			# N_k
			const = self.w[:, j].sum()
			#print (const)
			# phi_k=N_k/N eq.(19)
			self.phi[j] = 1/self.m * const
			# miu eq.(17)
			_mu_j = np.zeros(self.n)
			#sigma eq.(18)
			_sigma_j = np.zeros((self.n, self.n))
			for i in range(self.m):
				_mu_j += (self.w[i, j] * X_m_step[i, :])
				_sigma_j += self.w[i, j] * ((X_m_step[i, :] - self.mean_arr[j, :]).T * (X_m_step[i, :] - self.mean_arr[j, :]))
			self.mean_arr[j] = _mu_j / const
			self.sigma_arr[j] = _sigma_j / const
			# add the sigma_nk
			pre_n=-1
			for n in range (len(self.index_mi)):
				if pre_n!=self.index_mi[n]:
					pre_n=self.index_mi[n]
					missingcomponent=np.isnan(self.data[self.index_mi[n]])
					#提取self.sigma相应子矩阵部分更新
					#缺失掩膜
					index_=missingcomponent.reshape(len(missingcomponent),-1)*missingcomponent
					sigma=np.ma.MaskedArray(self.sigma_arr[j],mask=~index_)# == self.sigma_arr[j,missingcomponent][:,missingcomponent]
					prec=np.ma.MaskedArray(self.prec_arr[j],mask=~index_)# 精度矩阵
					d=sigma+1/const*(self.w[self.index_mi[n],j])*np.linalg.inv(prec)
					##d=sigma+1/const*(self.w[self.index_mi[n],j])*np.linalg.inv(np.asmatrix(self.prec_arr[j,missingcomponent][:,missingcomponent]))
					#b=1/const*(self.w[self.index_mi[n],j])*np.linalg.inv(np.asmatrix(self.prec_arr[j,missingcomponent][:,missingcomponent]))
					#print(b)
					#print(self.sigma_arr[j,missingcomponent][:,missingcomponent]+b)
					self.sigma_arr[j]=d
					#self.sigma_arr[j,missingcomponent][:,missingcomponent]=self.sigma_arr[j,missingcomponent][:,missingcomponent]+1/const*(self.w[self.index_mi[n],j])*np.linalg.inv(np.asmatrix(self.prec_arr[j,missingcomponent][:,missingcomponent]))
					#print (self.sigma_arr[j,missingcomponent][:,missingcomponent])
		print ('The m-step finished!')
		#print(self.sigma_arr)

	def plotdata(self):
		'''
		This function is used to plot example data
		'''
		data=self.data
		fig = plt.figure()

		x=data[:,0]
		y=data[:,1]

		# the first guassain
		data1=data[:200,]
		# the second guassain
		data2=data[200:,]

		x1 = data1[:,0]
		y1 = data1[:,1]

		x2 = data2[:,0]
		y2 = data2[:,1]

		plt.scatter(x1,y1,c='#FFA500',marker='o',alpha=0.6,label='x~N1(miu1,sigma1)')
		plt.scatter(x2,y2,c='#008000',marker='o',alpha=0.4,label='x~N2(miu2,sigma2)')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend()
		plt.show()
		print ('')

def R2(a, b): #a是模拟值，b是真实值
	a=np.array(a)
	b=np.array(b)
	r2=1 - ((a - b)**2).sum() / ((b - b.mean())**2).sum() 
	return r2
if __name__=="__main__":
	## To produce a data example
	#X = np.random.multivariate_normal([0, 3,1,10], [[5, 0,0,0], [0, 8,0,0],[0,0,1,0],[0,0,0,1]], 300)
	#X = np.vstack((X, np.random.multivariate_normal([1, 50,2,3], np.identity(4), 700)))
	#np.random.shuffle(X)

	## set X value randomly to zero
	#Xnan=X.copy()
	#ratioNan=0.3
	#num,dim=X.shape
	#Xration=np.random.rand(num,dim)
	#Xnan[Xration<ratioNan]=np.nan
	## remove the rows that have only NaN values
	#gg=np.isnan(Xnan).sum(axis=1)
	##find the index where the recored are all nans.
	#inde=np.where(gg==max(gg))
	#X=np.delete(X,gg==max(gg),axis=0)
		
	##Xnan2=np.delete(Xnan,gg==max(gg),axis=0)
	#XNAN=np.delete(Xnan,np.isnan(Xnan).sum(axis=1)==max(np.isnan(Xnan).sum(axis=1)),axis=0)

	#f2=open('data.csv','w',newline='')
	#fw2=csv.writer(f2)
	#fw2.writerows(X)
	#f2.close()
	#f3=open('missingdata.csv','w',newline='')
	#fw3=csv.writer(f3)
	#fw3.writerows(XNAN)
	#f3.close()

	# --------------------Option1 Example: Reading the missing data----------------------------------
	f=open('new_USC00042574_no_leap_days.csv','r')
	csv_read=csv.reader(f)
	data1=[line for line in csv_read]
	for l in range (len(data1)):
		for ll in range (len(data1[l])):
			data1[l][ll]=float(data1[l][ll])
	XNAN=np.array(data1,float)

	#------------------------------------------------------------------------------------------------

	##--------------------Option2 Data imputing: 读取气象数据-----------------------------------------
	#data=pd.read_csv('missing2.csv','r')
	#XNAN=np.array(data)
	#print(XNAN[0])
	##------------------------------------------------------------------------------------------------
	##标准化
	#scaler=StandardScaler()
	#scaler.fit(XNAN)
	#XNAN=scaler.transform(XNAN)
	#print ('Reading data is finished.')
	kk=6
	gmm=GMM(XNAN,kk) 
	#gmm.plotdata()
	gmm.fit()

	# save results.
	# save the latent variable Znk
	w=np.squeeze(gmm.neww)
	f5=open('Znk.csv','w',newline='')
	fw5=csv.writer(f5)
	for ww in w:
		a=ww.A1
		fw5.writerow(a)
	f5.close()

	## Evaluating the result.
	##Reading the TrueX
	##-----------------------Option 1 Example: Reading the complete data--------------------------------
	##f6=open('data.csv','r')
	##----------------------Option 2 Data imputing: 读取完整的气象数据----------------------------------------
	#f6=open('data2.csv','r')
	#csv_read2=csv.reader(f6)
	#data2=[line2 for line2 in csv_read2]
	#for l2 in range(len(data2)):
	#	for ll2 in range(len(data2[l2])):
	#		data2[l2][ll2]=float(data2[l2][ll2])
	#X=np.array(data2,float)

	#trueX=[]
	#imputingX=[]
	for i in range(len(gmm.index_mi)):
		#for j in [gmm.index_mj[i]]:
		#print (gmm.s[gmm.index_mi[i],j])
		#print(gmm.w[gmm.index_mi[i],:].A1)
		a=np.where(gmm.neww[gmm.index_mi[i],:].A1==max(gmm.neww[gmm.index_mi[i],:].A1))[0][0]
		print (a)
		XNAN[gmm.index_mi[i],gmm.index_mj[i]]=gmm.news[gmm.index_mi[i],gmm.index_mj[i]][a]
	###逆标准化
	##XNAN=scaler.inverse_transform(XNAN)
	#for i2 in range(len(gmm.index_mi)):
	#	trueX.append(X[gmm.index_mi[i2],gmm.index_mj[i2]])
	#	imputingX.append(XNAN[gmm.index_mi[i2],gmm.index_mj[i2]])
	f4=open('imputeddata.csv','w',newline='')
	fw4=csv.writer(f4)
	fw4.writerows(XNAN)
	f4.close()

	#with open('truex.csv','w') as f2:
	#	fw2=csv.writer(f2)
	#	fw2.writerow(trueX)
	#	f2.close()
	#with open('imputedX.csv','w') as f3:
	#	fw3=csv.writer(f3)
	#	fw3.writerow(imputingX)
	#	f3.close()
	#r,p=sp.stats.pearsonr(trueX,imputingX)
	##for ii in range (len(trueX)):
	##	print(float(trueX[ii])-float(imputingX[ii]))
	#r2=R2(trueX,imputingX)
	

	l_=np.linspace(1,len(gmm.newl),len(gmm.newl))
	plt.plot(l_,gmm.newl,'c*-')
	plt.xlabel('Time')
	plt.ylabel('ln LL')
	plt.title('The variation of ln-likelihood')
	plt.savefig('ln-likelihood.png',dpi=300)
	plt.clf()
	
	#plt.scatter(trueX,imputingX,c='c',marker='*')
	#plt.xlabel('True')
	#plt.ylabel('Imputed')
	#plt.title('The True and Imputed')
	#plt.savefig('Comparition.png',dpi=300)

	with open('GMM.txt','w') as ff:

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

		#ff.write('The correlation is =%f'%r)
		#ff.write ('The p is =%f'%p)
		#ff.write ('The r2 is =%f'%r2)
		#ff.write('\n')
		#ff.write('The R2 is =%f'%r2)
	ff.close()



