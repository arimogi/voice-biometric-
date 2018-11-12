import csv, random
import pandas as pd
import numpy as np
import math
import operator
from sklearn import preprocessing



with open('wavelet.csv') as csvfile :
    readCSV = csv.reader(csvfile, delimiter=',')
    df1=pd.read_csv("wavelet.csv")
    print(df1)
    newar=[];
    newar1=[];
    a = 0;
    label =0
    for row in readCSV:
		#print (row)
		#if row[41]=="neptune." or row[41]=="smurf." or row[41]=="normal."

			#data_ = [float(row[2]), float(row[5]), float(row[23]), float(row[24]), row[31], row[32], row[33], row[37], row[39]]
        # data1 = [float(row[2])]
        data2 = [float(row[3])]
        # newar.append(data1)
        newar1.append(data2)
		
    # data1 = newar
    data2 = newar1
	
