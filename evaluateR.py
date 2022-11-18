#-*- coding=utf-8 -*-
'''This code is used to evaluate the imputing performance by R2 and pearson's r.
'''
import csv
import numpy as np
import scipy as sp
from scipy import stats
def R2(a, b): #a是模拟值，b是真实值
	a=np.array(a)
	b=np.array(b)
	r2=1 - ((a - b)**2).sum() / ((b - b.mean())**2).sum() 
	return r2

with open ('data2.csv','r') as f1:
	'''
	open the true data.
	'''
	fr1=csv.reader(f1)
	data=[l for l in fr1]
data=np.array(data).astype(np.float)

with open('imputeddata.csv','r') as f2:
	'''
	Open the imputed data. 
	'''
	fr2=csv.reader(f2)
	imputed=[ll for ll in fr2]
imputed=np.array(imputed).astype(np.float)
difference=imputed - data
index=np.nonzero(difference)
column=set(index[-1])
row=set(index[0])
n=0
for i in column:
	di=[]
	dt=[]
	rowindex=np.where(index[-1]==i)[0]
	print(len(rowindex))
	for r in rowindex:
		di.append(imputed[index[0][r]][i])#imputed data
		dt.append(data[index[0][r]][i])#true data
	#with open('x1_i%i.csv'%n,'w',newline='') as f3:
	#	fw3=csv.writer(f3)
	#	fw3.writerow(di)
	#with open('x1_t%i.csv'%n,'w',newline='') as f4:
	#	fw4=csv.writer(f4)
	#	fw4.writerow(dt)
	
	n=n+1
	r2=R2(di,dt)
	print ('Rsquare=%f'%r2)
	r,p=stats.pearsonr(di,dt)
	print ('correlation =%f and the p value is %f'%(r,p))
print ('')
