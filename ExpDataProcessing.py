from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from numpy import pi, sin, cos, sqrt
import numpy as  np

Degree= pi/180

alpha12 = 58.4*Degree
alpha13 = 121.6*Degree
alpha23 = 180*Degree

def weight_function(alpha):
    ''' Calculating the bias parameter \( omega\) from the relative angle between
    the target pair of Bloch vectors \( \cos(\alpha) = \vec{n}_i \dot \vec{n}_j  \)'''
    return 0.5*(1.0 + (1.0 - abs(sin(alpha)))*(1./cos(alpha)))

service = QiskitRuntimeService()


with open("JobID.txt", "r") as file:
    job_id=file.read()


job = service.job(job_id.strip())


shots=4096

result = job.result()

#  Retrieving counts
counts=[]
num_of_circuits = 3*2*3 
for  i in range(num_of_circuits):
        dist = result[i].data.c.get_counts()
        counts.append(dist)


# Calculating the associated expectation values
E=[]

#List of expectation values
for count in counts :
    evs=(count["0"]-count["1"])/shots
    E.append(evs)

# List (of dictionaries) of outcome probabilities 
proba=counts

for i in range(num_of_circuits):
    proba[i]["0"] = proba[i]["0"]/shots;
    proba[i]["1"] = proba[i]["1"]/shots

# Variances of the expectation values 
Var=[]
for i in range(num_of_circuits):
    Var.append(4*proba[i]["0"]*(1-proba[i]["0"])/shots)

# Calculating witness I3(\omgea_{12})

Exy =  np.array([[E[0], E[1]],
                 [E[2], E[3]],
                 [E[4], E[5]]] );

Varxy = np.array([[Var[0],  Var[1]],
                  [Var[2],  Var[3]],
                  [Var[4],  Var[5]]] );

w=weight_function(alpha12)

Wxy= np.array([[w,   (1-w)],
               [w,  -(1-w)],
               [-w,  0]]);

I_12=np.sum(Exy*Wxy)

# standard deviation 
Var12 = np.sum(Wxy**2*Varxy)
delta12=sqrt(Var12)

print(I_12)

# Calculating witness I3(\omgea_{13})

Exy =  np.array([[E[6], E[7]],
                 [E[8], E[9]],
                 [E[10], E[11]]] );

Varxy = np.array([[Var[6],  Var[7]],
                  [Var[8],  Var[9]],
                  [Var[10], Var[11]]] );

w=weight_function(alpha13)

Wxy= np.array([[w,   (1-w)],
               [w,  -(1-w)],
               [-w,  0]]);

I_13=np.sum(Exy*Wxy)


# standard deviation 
Var13 = np.sum(Wxy**2*Varxy)
delta13=sqrt(Var13)

print(I_13)

# Calculating witness I3(\omgea_{23})

Exy =  np.array([[E[12], E[13]],
                 [E[14], E[15]],
                 [E[16], E[17]]] );

Varxy = np.array([[Var[12],  Var[13]],
                  [Var[14],  Var[15]],
                  [Var[16],  Var[17]]] );

w=weight_function(alpha23)

Wxy= np.array([[w,   (1-w)],
               [w,  -(1-w)],
               [-w,  0]]);

I_23=np.sum(Exy*Wxy)


# standard deviation 
Var23 = np.sum(Wxy**2*Varxy)
delta23=sqrt(Var23)

print(I_23)

# Overall witness
Q = I_12 + I_13 + I_23

# standard deviation
delta = sqrt( delta12**2+delta13**2+delta23**2)

print(Q, delta)
