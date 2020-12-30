fin = open("TrainingData.csv")
x = open("x.csv", 'w')
Y = open("Y.csv", 'w')

xData = []
yData = []

# read in data
for dataSet in fin:
    d = list(map(int,dataSet.strip().split(',')))
    xData.append(d[0:3])
    yData.append([d[-1]])

# write data
index = 0
for i in xData:
    k = str(i[0])
    for j in i[1:]:
        k += ',' + str(j)
    index+=1
    x.write(k+'\n' if index != len(xData) else k+'\n')

for i in yData:
    k = ('0,1' if i[0] == 1 else '1,0')
    index+=1

    Y.write(k+'\n' if index != len(yData) else k+'\n')

x.close()
Y.close()
fin.close()
