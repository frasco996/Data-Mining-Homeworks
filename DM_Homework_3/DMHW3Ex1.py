import copy 
import random
random_list = [random.randint(1,2000) for _ in range(50)]
#random_list.sort()


dict = {'centroid':0.0,'npoints':0.0,'cost':0.0,'listpoints':[]}

centroid = sum(random_list)/len(random_list)
cost = sum([(x**2 - centroid**2) for x in random_list])
dict = {'centroid':centroid,'npoints':len(random_list),'cost':cost,'listpoints':random_list}
listdict = [dict]

def separate(parlistdict):
    idxmax = 0 
    costomax = 0.0
    for idx , d in enumerate(parlistdict):
        if d['cost'] > costomax and d['npoints'] > 1:
            costomax = d['cost']
            idxmax = idx
    
    pardict = parlistdict[idxmax]
    
    
    cs = [x for x in pardict['listpoints'] if x <= pardict['centroid']]
    cd = [x for x in pardict['listpoints'] if x > pardict['centroid']]
    
    centroidCs = sum(cs)/len(cs)
    centroidCd = sum(cd)/len(cd)
    
    costCs = sum([(x**2 - centroidCs**2) for x in cs])
    costCd = sum([(x**2 - centroidCd**2) for x in cd])
    
    dictCs = {'centroid':centroidCs,'npoints':len(cs),'cost':costCs,'listpoints':cs}
    dictCd = {'centroid':centroidCd,'npoints':len(cd),'cost':costCd,'listpoints':cd}
    
    parlistdictcopy = copy.deepcopy(parlistdict)
    parlistdictcopy[idxmax] = dictCs
    parlistdictcopy.insert(idxmax+1 , dictCd)
    
    #return [dictcs , dictcd]
    if len(parlistdictcopy) >= 5:
        return parlistdictcopy
    else:
        return separate(parlistdictcopy)



listCluster = separate(listdict)

#DECREMENT COST
costnew = 0.0
for c in listCluster:
    costnew += c['cost']

print(listCluster)

diff = cost - costnew