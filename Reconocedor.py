import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from random import randint

def logsigmoid(wab):
	F = np.zeros((wab.shape[0],wab.shape[1]))
	for t in range(0,wab.shape[0]):
		for l in range(0,wab.shape[1]):
			F[t,l] = 1 / (1 + math.exp(-wab[t,l]))
	return F
##################
def diagonal(a):
	df1 = np.zeros((a.shape[0],a.shape[0])) 
	for t in range(0,a.shape[0]):
		df1[t,t] = (1-a[t,0])*a[t,0]
	return df1
#####################
def algoritmo(entradas,neuronas,p,t):
	w1 = np.random.rand(neuronas,entradas)-0.5
	b1 = np.random.rand(neuronas,1)-0.5
	w2 = np.random.rand(4,neuronas)-0.5
	b2 = np.random.rand(4,1)-0.5
	for j in range(0,10000):
		##Modificar dependiendo de que datos se resiven
		for q in range(0,40):##Con datos completos
		#for q in range(0,32):##Con 80% de aprendizaje
			a0 = np.zeros((1,16))
			te = np.zeros((1,4))
			for v in range(0,16):
				a0[0,v] = p[v,q]
			#####################
			a0 = a0.T
			f = np.dot(w1,a0) + b1
			a1 = logsigmoid(f)
			f = np.dot(w2,a1) + b2
			a2 = logsigmoid(f)
			for v in range(0,4):
				te[0,v] = t[v,q]
			#####################
			te = te.T
			error = te - a2
			df1 = diagonal(a1)
			df2 = diagonal(a2)
			s2 = -2*np.dot(df2,error)
			s1 = np.dot(df1,w2.T)
			s1 = np.dot(s1,s2)
			alfa = 0.001
			w2 = w2 - alfa*np.dot(s2,a1.T)
			b2 = b2 - alfa*s2
			w1 = w1 - alfa*np.dot(s1,a0.T)
			b1 = b1 - alfa*s1			
	return [w2,b2,w1,b1]
#####################

archivo = open("Datos.txt")
x = []
for line in archivo.read():
	y=[v for v in line.split()]
	if y!=[ ]:
		x.append(int(y[0]));

datosA0 = np.zeros((32,16))
datosP0 = np.zeros((8,16))
datosT = np.zeros((40,16))
t = 0
n = 1
for i in range(0,40):
	if ((i == 9) or (i == 19) or (i == 29) or (i == 39)):
		n = n + 1
	for j in range(0,16):
		d = x[t]
		if((0 <= i < 8) or ( 10 <= i < 18) or ( 20 <= i < 28) or ( 30 <= i < 38)):
			datosA0[i-2*(n-1),j] = d
		else:
			datosP0[i-8*n,j] = d

		datosT[i,j] = d
		t = t + 1
datosA0 = datosA0.T##Datos de entrenamiento
datosP0 = datosP0.T##Datos de prueba
datosT = datosT.T##Datos completos


archivo.close() 
total = 10
entrena = 8
###Modifica para la cantidad de datos
cantidad = total ##Para datos completos
#cantidad = entrena ###Para entrenamiento 80%
T = np.array(np.ones((1,cantidad)))
T = np.concatenate((T,np.zeros((3,cantidad))))
for i in range(0,3):
	N = np.array(np.zeros((i+1,cantidad)))
	N = np.concatenate((N,np.ones((1,cantidad))))
	N = np.concatenate((N,np.zeros((2-i,cantidad))))
	T =  N = np.concatenate((T,N),1)


neuronas = 5
entradas = 16
###Se puede pasar los datos completos o solo los de entranamiento para despues 
###poder evaluar el aprendizaje.
#[w2,b2,w1,b1] = algoritmo(16,neuronas,datosA0,T)##80% Para aprendizaje
[w2,b2,w1,b1] = algoritmo(16,neuronas,datosT,T)##Datos completos

print("\nw1:\n",w1)
print("\nb1:\n",b1)
print("\nW2:\n",w2)
print("\nb2:\n",b2)
print("\n")
archivo = open("Buscar.txt")
x = []
for line in archivo.read():
	y=[v for v in line.split()]
	if y!=[ ]:
		x.append(int(y[0]));
		#############
busca = np.zeros((1,16))
for j in range(0,np.size(x)):
		d = x[j]
		busca[0,j] = d
busca = busca.T
##############################
for r in range(0,1):###Modificar dependiendo que datos se analize datos total '8'
					###datos para probar '1'
	a0 = np.zeros((1,16))
	for v in range(0,16):
		a0[0,v] = busca[v,0]##Probar de archivo
		#a0[0,v] = datosP0[v,r]##Probar datos 20%

	#####################
	a0 = a0.T
	f = np.dot(w1,a0)  + b1
	a1 = logsigmoid(f)
	f = np.dot(w2,a1) + b2
	a2 = logsigmoid(f)

	recono = 0###lugar de la mayor ganancia
	for i in range(0,a2.shape[0]):
		if(max(a2) == a2[i,0]):		
			recono = i
	print("El valor es: ", recono)

archivo.close() 