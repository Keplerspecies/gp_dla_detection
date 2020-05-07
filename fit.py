import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys

D = np.loadtxt(sys.argv[1])

D = D[:,np.isfinite(D[1,:])]

#npts = 1000
#truemu = 2
#truevar = 4
#vs = np.random.rand(1,npts)*25
#D = np.vstack((np.random.randn(1,npts)*np.sqrt(vs+truevar)+truemu,vs))
m = D.shape[1]

spread = 100

def fitmu(D,sigma2,dens=None):
    if dens is None:
        dens = D[1,:] + sigma2
    return np.sum(D[0,:]/dens)/np.sum(1.0/dens)

# ignores the sqrt(2pi) part
def negllh(D,sigma2,mu=None):
    dens = D[1,:] + sigma2
    if mu is None:
        mu = fitmu(D,sigma2,dens)
    return np.sum(np.log(dens) + np.square(D[0,:] - mu)/dens)/2

naivemu = np.mean(D[0,:])
naivevar = np.sum(np.square(D[0,:]-naivemu))/m - np.mean(D[1,:])

onaivemu = np.sum(D[0,:]/D[1,:])/np.sum(1.0/D[1,:])
onaivevar = np.sum(np.square(D[0,:]-onaivemu)/D[1,:])/np.sum(1.0/D[1,:]) 

'''
print("ranges: ",np.max(D[0,:]),np.min(D[0,:]),np.max(D[1,:]),np.min(D[1,:]))
print ("mean var: ",np.mean(D[1,:]))
print(naivemu,naivevar,negllh(D,naivevar,naivemu))
print(onaivemu,onaivevar,negllh(D,onaivevar,onaivemu))
'''

lessnaivemu = fitmu(D,naivevar)
#print(lessnaivemu,naivevar,negllh(D,naivevar,lessnaivemu))
olessnaivemu = fitmu(D,onaivevar)
#print(olessnaivemu,onaivevar,negllh(D,onaivevar,olessnaivemu))

# REAL WORK NEXT LINE:
optsol = opt.minimize_scalar(lambda sigma2 : negllh(D,sigma2), bounds=(onaivevar/spread,onaivevar*spread),method='bounded')

if not optsol.success:
    print("optimization failed")
optvar = optsol.x
optmu = fitmu(D,optvar)
#print('optmu, optvar, negllh: ');
print(optmu,optvar,negllh(D,optvar,optmu))
'''
xs = np.logspace(np.log(naivevar/spread),np.log(naivevar*spread),100)
ys = np.array([negllh(D,x) for x in xs])
plt.semilogx(xs,ys)
plt.semilogx(optvar,negllh(D,optvar,optmu),'k*')
plt.semilogx(naivevar,negllh(D,naivevar,naivemu),'g*')
plt.semilogx(naivevar,negllh(D,naivevar),'c*')
plt.semilogx(onaivevar,negllh(D,onaivevar,onaivemu),'r*')
plt.semilogx(onaivevar,negllh(D,onaivevar),'m*')
plt.legend(['negllh','opt value, mu=%g' % optmu,'naive mu & var, mu=%g' % naivemu,'naive var, opt mu, mu=%g' % lessnaivemu,'2nd naive mu & var, mu=%g' % onaivemu,'2nd naive var, opt mu, mu=%g' %olessnaivemu])
plt.xlabel('variance')
plt.ylabel('negative log-likelihood')
plt.show()
'''
