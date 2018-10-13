import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from random import randint

pi = 3.1416
x = np.arange(-2,2,0.1)/1
gp = 1 + np.sin((1*pi*x)/2)
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

def algoritmo(entradas,neuronas,rate):
	w1 = np.random.rand(neuronas,entradas)-0.5
	b1 = np.random.rand(neuronas,entradas)-0.5
	w2 = np.random.rand(4,neuronas)-0.5
	b2 = np.random.rand(4,entradas)-0.5
	w3 = np.random.rand(entradas,4)-0.5
	b3 = 0.48
	for j in range(0,10000):
		p = randint(-2,2)		
		a0 = p
		f = w1*a0 + b1
		a1 = logsigmoid(f)
		f = np.dot(w2,a1) + b2
		a2 = logsigmoid(f) 
		f = np.dot(w3,a2) + b3
		a3 = purelim(f) 
		t = 1 + np.sin((pi*p)/2)
		error = t - a3			
		df3 = 1
		s3 = -2*df3*error
		df1 = diagonal(a2)		
		s2 = np.dot(df1,w3.T)*s3
		df1 = diagonal(a1)
		s1 = np.dot(np.dot(df1,w2.T),s2)
		alfa = rate
		w3 = w3 - alfa*s3*a2.T
		b3 = b3 - alfa*s3
		w2 = w2 - alfa*s2*a1.T
		b2 = b2 - alfa*s2
		w1 = w1 - alfa*s1*a0
		b1 = b1 - alfa*s1	
	gp2 = np.zeros((1,np.size(x)))
	for i in range(0,np.size(x)):
		f = w1*x[i] + b1
		a1 = logsigmoid(f)
		f = np.dot(w2,a1) + b2
		a2 = logsigmoid(f)
		f = np.dot(w3,a2) + b3
		a3 = purelim(f)
		gp2[0,i] = a3
	return gp2
###################




plt.figure(1)
plt.plot(x,gp,'*')
plt.title('a) S = 2 y Learning Rate = 0.5')
gp2 = algoritmo(1,2,0.5)
plt.plot(x,gp2[0,:])
plt.show()

plt.figure(2)
plt.plot(x,gp,'*')
plt.title('b) S = 2 y Learning Rate = 1')
gp2 = algoritmo(1,2,1)
plt.plot(x,gp2[0,:])
plt.show()

plt.figure(3)
plt.plot(x,gp,'*')
plt.title('c) S = 10 y Learning Rate = 0.5')
gp2 = algoritmo(1,10,0.5)
plt.plot(x,gp2[0,:])
plt.show()

