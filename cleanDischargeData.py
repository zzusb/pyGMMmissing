from cmath import nan
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import  ConnectionPatch
'''
This is used to clean discharge data of USA.
'''

def Cleanstreamflow(path,sheet):
	'''
	Used to clean the discharge data.
	'''
	#path=r"C:\Users\Mason\Desktop\discharge.csv"
	#df=pd.read_csv(path,parse_dates=["Date"])

	df=pd.read_excel(path,sheet_name=sheet,header=None,names=["Date","Discharge"],skipfooter=0,parse_dates=["Date"])
	streamflow=[]
	for d in df["Discharge"]:
		d=d.replace(u'\xa0', u'')
		d=d.strip('A??e')
		d=d.strip('P??e')
		d=d.strip('P??')
		dd=d.strip("A??")
		ddd=dd.replace(',','')
		if ddd=="":
			dddd=nan
		else:
			dddd=float(ddd)*0.02831685
		streamflow.append(dddd)
	df["streamflow_m3_s_1"]=streamflow
	avg=df["streamflow_m3_s_1"].mean(axis=0)
	del df["Discharge"]
	df.to_csv(r"F:/Hydrology/discharge/cleaned/discharge_%s.csv"%sheet,index=None)
	plt.plot(df["Date"],df["streamflow_m3_s_1"],linewidth=1,color="#0623F5")
	plt.title("Discharge_%s: Avg=%.2f"%(sheet,avg))
	plt.xlabel("Daily")
	plt.ylabel("streamflow(m\u00b3/s)")
	#plt.show()
	plt.savefig(r"F:/Hydrology/discharge/cleaned/discharge_%s.png"%sheet,dpi=900)
	plt. clf()




if __name__=="__main__":
	#---------------------clean the discharge data.----------------------------------------
	discharge_id=[11342000,11368000,11367500,11365000,11348500,11345500,11351700]
	path=r"F:/Hydrology/discharge/discharge.xlsx"
	for i in discharge_id:
		sheet=str(i)
		Cleanstreamflow(path,sheet)
		print ("Discharge data are cleaned.")
	#---------------------------------------------------------------------------------------
	print ("")

