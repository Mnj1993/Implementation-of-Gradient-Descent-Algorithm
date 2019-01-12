import sys
import math
import random
################
##  py GradientDescent.py datafile trainlabelfile etaValue
################

#############
## Sub routines
#############
def dotproduct(x, y):
    if(len(x) == len(y)):
        dp = 0
        for i in range(0, len(x), 1):
            dp += x[i]*y[i]
    return dp

################
##Read Data
################
datafile = sys.argv[1]
f = open(datafile)
data = []
i=0
l=f.readline()
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    l2.append(1)
    data.append(l2)
    l=f.readline()

rows = len(data)
cols = len(data[0])
f.close()

################
##Read Labels
################
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l != ''):
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    if(trainlabels[int(a[1])] == 0):
        trainlabels[int(a[1])] = -1
    l = f.readline()
    n[int(a[0])] += 1

############
## Read eta
############
eta = float(sys.argv[3])

############
##Initialize w
############ 
w=[]
for i in range(0,cols):
    w.append(0.002*random.random()-0.001)

##################
##Gradient descent iteration
##################

#eta = 0.0001
dellf = []
for j in range(0, cols, 1):
    dellf.append(0)

lastObjective = 10000000
objective = lastObjective - 10

while(lastObjective - objective > eta):
    lastObjective = objective
    for j in range(0, cols, 1):
        dellf[j] = 0
    
    ## Computer dellf ##
    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            dp = dotproduct(w, data[i])
            for j in range(0, cols, 1):
                dellf[j] = dellf[j] + ((trainlabels.get(i) - dp)*data[i][j])

    for j in range(0, cols, 1):
        w[j] = w[j] + (eta * dellf[j])

    ## Calculating error ##
    error = 0
    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            error = error + (trainlabels.get(i) - dotproduct(w, data[i]))**2
    objective = error
    print("Objective is : ", error)

print("w: ", w)
normw = math.sqrt(w[0]**2 + w[1]**2)
d_origin = abs(w[2])/normw
print("Distance to origin: ", d_origin)

###########################
## Clasify unlabeled points
###########################

for i in range(0, rows, 1):
    if(trainlabels.get(i) == None):
        dp = dotproduct(w, data[i])
        if (dp > 0):
            print("1 ", i)
        else:
            print("0 ", i)
