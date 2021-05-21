import pandas as pd
from scipy import stats
import numpy as np
import math as m
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rcParams.update({'font.size': 25})

df = pd.read_csv('Dados90.csv')

VEmod=df['Vemod'][399]
VEfase=(df['Vefase'][399]*m.pi/180)
IEmod=df['Iemod'][399]
IEfase=(df['Iefase'][399]*m.pi/180)
VSmod=df['Vsmod'][399]
VSfase=(df['Vsfase'][399]*m.pi/180)
ISmod=df['Ismod'][399]
ISfase=(df['Isfase'][399]*m.pi/180)

n=5000

sigma = 0.02

thetamax =8*(10 ** (-3))

B=np.ones(n)

np.random.seed(42)
w1=VEmod*np.ones(n)+VEmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(41)
w2=VEfase+(thetamax/3)*np.random.normal(0,1,n)
np.random.seed(40)
w3=IEmod*np.ones(n)+IEmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(39)
w4=IEfase+(thetamax/3)*np.random.normal(0,1,n)
np.random.seed(38)
w5=VSmod*np.ones(n)+VSmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(37)
w6=VSfase+(thetamax/3)*np.random.normal(0,1,n)
np.random.seed(36)
w7=ISmod*np.ones(n)+ISmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(35)
w8=ISfase+(thetamax/3)*np.random.normal(0,1,n)

realIe=np.ones(n)
imVe=np.ones(n)
imIe=np.ones(n)
realVe=np.ones(n)
realIs=np.ones(n)
imVs=np.ones(n)
imIs=np.ones(n)
realVs=np.ones(n)

for j in range(0,n):
    realIe[j]=w3[j]*np.cos(w4[j])
    imVe[j]=w1[j]*np.sin(w2[j])
    imIe[j]=w3[j]*np.sin(w4[j])
    realVe[j]=w1[j]*np.cos(w2[j])
    realIs[j]=w7[j]*np.cos(w8[j])
    imVs[j]=w5[j]*np.sin(w6[j])
    imIs[j]=w7[j]*np.sin(w8[j])
    realVs[j]=w5[j]*np.cos(w6[j])

for l in range(0,n):
    B[l]=-2*((realIe[l]+realIs[l])/(imVe[l]+imVs[l]))

data={'realIe':realIe,'realIs':realIs,'realVe':realVe,'realVs':realVs,'imIe':imIe,'imIs':imIs,'imVe':imVe,'imVs':imVs,'B':B}

df2=pd.DataFrame(data)

df2.to_csv('file.csv')

shapiro_testRealIe = stats.shapiro(realIe)

shapiro_testRealIs = stats.shapiro(realIs)

shapiro_testRealVe = stats.shapiro(realVe)

shapiro_testRealVs = stats.shapiro(realVs)

shapiro_testImIe = stats.shapiro(imIe)

shapiro_testImIs = stats.shapiro(imIs)

shapiro_testImVe = stats.shapiro(imVe)

shapiro_testImVs = stats.shapiro(imVs)

shapiro_testB = stats.shapiro(B)

##### coeficientes de Person entre cada parte real e imaginária ####

rhoIe=np.corrcoef(realIe, imIe)
rhoIs=np.corrcoef(realIs, imIs)
rhoVe=np.corrcoef(realVe, imVe)
rhoVs=np.corrcoef(realVs, imVs)

print("Valores obtidos para o teste de Shapiro-Wilk Normalidade \n")
print(shapiro_testRealIe,'\n')
print(shapiro_testRealIs,'\n')
print(shapiro_testRealVe,'\n')
print(shapiro_testRealVs,'\n')

print(shapiro_testImIe,'\n')
print(shapiro_testImIs,'\n')
print(shapiro_testImVe,'\n')
print(shapiro_testImVs,'\n')

print(shapiro_testB, '\n')

print('Valores obtidos para os coeficientes de Person entre cada parte real e imaginária \n')

print(rhoIe,'\n')
print(rhoIs,'\n')
print(rhoVe,'\n')
print(rhoVs,'\n')

axis_font = {'fontname':'Arial', 'size':'25'}

n, bins, patches = plt.hist(x=imIe, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',**axis_font)
plt.ylabel('Absolute Frequency',**axis_font)
plt.title(' Histogram for the real part of the current $\dot{I}_e$')
plt.show()

sm.qqplot(imIe,line='s')
plt.show()

