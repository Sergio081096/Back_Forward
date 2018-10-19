import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from random import randint
w1 = np.array([[-0.27],[-0.41]])
b1 = np.array([[-0.48],[-0.13]])
w2 = np.array([[0.09,-0.17]])
b2 = 0.48
pi = 3.1416
x = np.arange(-20,20,1)/10
gp = 1 + np.sin((pi*x)/4)
global gp2 
global df1
def logsigmoid(wab):
	F = np.zeros((wab.shape[0],1))
	for t in range(0,wab.shape[0]):
		F[t,0] = 1 / (1 + math.exp(-wab[t,0]))
	return F
##############################
def purelim(wab):
	wab = wab[0,0]
	return wab
##################
def diagonal(a):
	df1 = np.zeros((a.shape[0],a.shape[0])) 
	for t in range(0,a.shape[0]):
		df1[t,t] = (1-a[t,0])*a[t,0]
	return df1
#####################
def algoritmo(p,w1,b1,w2,b2,i):
	for j in range(0,i):		
		a0 = p### P toma el valor inicial despues de vuelve aleatorio
		f = w1*p + b1
		a1 = logsigmoid(f) 
		f = np.dot(w2,a1) + b2
		a2 = purelim(f)
		t = 1 + np.sin((pi*p)/4)
		error = t - a2
		df1 = diagonal(a1)
		df2 = 1
		s2 = -2*df2*error
		s1 = np.dot(df1,w2.T)*s2
		alfa = 0.1
		w2 = w2 - alfa*s2*a1.T
		b2 = b2 - alfa*s2
		w1 = w1 - alfa*s1*a0
		b1 = b1 - alfa*s1
		if(i == 2):
			p = -1
		else:
			p = randint(-2,2)	
	gp2 = np.zeros((1,np.size(x)))
	for i in range(0,np.size(x)):
		f = w1*x[i] + b1
		a1 = logsigmoid(f)
		f = np.dot(w2,a1) + b2
		a2 = purelim(f)
		gp2[0,i] = a2 
	return gp2
#################################



plt.figure(1)
plt.subplot(1,2,1)
plt.plot(x,gp,'*')
plt.title('1° Iteración')
gp2 = algoritmo(1,w1,b1,w2,b2,1)
plt.plot(x,gp2[0,:])
plt.subplot(1,2,2)
plt.plot(x,gp,'*')
plt.title('2° Iteración')
gp2 = algoritmo(1,w1,b1,w2,b2,2)
plt.plot(x,gp2[0,:])
plt.show()

plt.figure(2)
plt.plot(x,gp,'*')
plt.title('Resultado Final')
gp2 = algoritmo(1,w1,b1,w2,b2,10000)
plt.plot(x,gp2[0,:])
plt.show()