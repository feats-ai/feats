import numpy as np
from operator import add 

def line2arr(x):
    x = x.split()
    
    for i in range(2):
        x[i] = int(x[i])
    for i in range(2,5):
        x[i] = float(x[i])
    return x



file = open('sim.dat', 'r')

for x in file:
    if len(x.split()) > 0:
        if x.split()[0] == 'contact':
            break

file.readline()
x = file.readline()
x = line2arr(x)
forcesPerContactElement = [x] 


for x in file:
    if x == '\n':
        break
    x = line2arr(x)
    if x[0] != forcesPerContactElement[-1][0] or x[1] != forcesPerContactElement[-1][1]:
        forcesPerContactElement += [x]
    else:
        for i in range(2,5):
            forcesPerContactElement[-1][i] += x[i] 

file.close()

sum = 0
for x in forcesPerContactElement:
    print(x)
    sum+=x[4]
print("Total Force on Contact Surface: {}".format(sum))
