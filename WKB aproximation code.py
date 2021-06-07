import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy as scp
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import scipy.linalg as sslin
from sympy import Array
from sympy import derive_by_array
from sympy import *
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sympy.plotting import plot3d
import cmath

# We start with Schloegl's second model
a=0.01
K=0.21
p,q= symbols('p q')
rateout=q**3 + K*q #rate outgoing
ratein=q**2 + a #rate incoming
# Define the Hamiltonian
Hfun= (exp(p)-1)*ratein+(exp(-p)-1)*rateout
Hfundp = lambdify([q,p],diff(Hfun,p))
Hfundq = lambdify([q,p],diff(Hfun,q))
# Equilibria
equilibria = np.sort(solve(rateout-ratein,q))

# Plot vector field
x,y = np.meshgrid(np.linspace(0,1,15),np.linspace(-0.17,0.15,10))
u = Hfundp(x,y)
v = -Hfundq(x,y)

fig, ax = plt.subplots()

ax.streamplot(x,y,u,v, density = 1.7, color='grey')
x1 = np.linspace(0.03,0.95,100)
plt.plot(x1,np.log((x1**3+K*x1)/(x1**2+a)), linewidth=3)
x1 = np.linspace(0,1,100)
plt.plot(x1,x1*0,color='red', linewidth=3)
plt.plot(equilibria, equilibria*0,'o', color='black', markersize=8)
plt.plot(0.2, 0,'o', color='black', mfc='white', markersize=8)
plt.xlabel('Concentration')
plt.ylabel('P')
plt.ylim((-0.17,0.15))
plt.xlim((0,1))

# We will start by using the system using the first order WKP app:

def modelsimfun(x1,Hfun,equilibria,T,Nt,eps,uplimit,downlimit,eqeps):
    p,q = symbols('p q')
    Hfundp = lambdify([q,p],diff(Hfun,p))
    Hfundq = lambdify([q,p],diff(Hfun,q))
    eq=np.setdiff1d(equilibria,x1)
    def RHS(y,t):
        dy = np.zeros(3)
        dy[0] = Hfundp(y[0],y[1]) #q
        dy[1] = -Hfundq(y[0],y[1]) #p
        dy[2] = y[1]*dy[0]
        if dy[1] > 0 and y[1] > uplimit:
            y[1] = uplimit
            dy[1]=0
            dy[0]=0
            dy[2]=0
        if dy[1] < 0 and y[1] < downlimit:
            y[1] = downlimit
            dy[1]=0
            dy[0]=0
            dy[2]=0
        for i in range(len(eq)):
            if np.abs(y[0]-eq[i]) < eqeps:
                if y[1]<eqeps:#could probably get rid of this if
                    y[0]=eq[i]
                    dy[0]=0
                    y[1]=0
                    dy[1]=0
                    dy[2]=0
                    break
        return dy
    y0 = np.zeros(3)
    y0[0] = x1
    y0[1] = eps
    y0[2] = 0
    t = np.linspace(0,T,Nt)
    #compute solution
    y = odeint(RHS,y0,t)
    q = y[:,0]
    p = y[:,1]
    s = y[:,2]
    return(q,p,s)
x2=0.7315
x1=0.0683
downlimit=-100
qr1,pr1,sr1=modelsimfun(x1,Hfun,equilibria,194,100000,-1e-8,0.15,downlimit,1e-10)
qr2,pr2,sr2=modelsimfun(x1,Hfun,equilibria,420,100000,1e-10,0.15,downlimit,1e-10)
qr3,pr3,sr3=modelsimfun(x2,Hfun,equilibria,150,100000,-1e-10,0.15,downlimit,1e-10)
qr4,pr4,sr4=modelsimfun(x2,Hfun,equilibria,60,100000,1e-10,0.3,downlimit,1e-10)

plt.plot(qr1,sr1)
plt.plot(qr2,sr2)
plt.plot(qr3,sr3)
plt.plot(qr4,sr4)
plt.title('Quasipotential')
plt.ylabel(r'$\bar{S}(q)$')
plt.xlabel(r'$q$')
#This looks just as we expected
# Now we want to match s2 and s3
c=sr3[-1]-sr2[-1]
srn1=sr1+c
srn2=sr2+c

plt.plot(qr1,srn1)
plt.plot(qr2,srn2)
plt.plot(qr3,sr3)
plt.plot(qr4,sr4)
# plt.title('Quasipotential')
plt.ylabel(r'$\bar{S}(q)$')
plt.xlabel('Concentration')
plt.annotate('$x_s$', xy=(0.31,0.73),xycoords='figure fraction', fontsize=15)
plt.annotate('$x_0$', xy=(0.2,0.54),xycoords='figure fraction', fontsize=15)
plt.annotate('$x_0\'$', xy=(0.63,0.22),xycoords='figure fraction', fontsize=15)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.plot(equilibria, [srn1[0],srn2[-1],sr3[0]],'o', color='black', markersize=6)



x2=0.7315
x1=0.0683
qr1,pr1,sr1=modelsimfun(x1,Hfun,equilibria,184,100000,-1e-8,0.15,-0.17,1e-10)
qr2,pr2,sr2=modelsimfun(x1,Hfun,equilibria,420,100000,1e-10,0.15,-0.17,1e-10)
qr3,pr3,sr3=modelsimfun(x2,Hfun,equilibria,150,100000,-1e-10,0.15,-0.17,1e-10)
qr4,pr4,sr4=modelsimfun(x2,Hfun,equilibria,58,100000,1e-10,0.15,-0.17,1e-10)
plt.plot(qr1,pr1, linewidth=3)
plt.plot(qr2,pr2, linewidth=3)
plt.plot(qr3,pr3, linewidth=3)
plt.plot(qr4,pr4, linewidth=3)
x1 = np.linspace(0.03,0.95,100)
plt.plot(x1,np.log((x1**3+K*x1)/(x1**2+a)), linewidth=1.5, color='white')
plt.plot(equilibria, equilibria*0,'o', color='black', markersize=6)
plt.ylabel('p')
plt.xlabel('Concentration')
xrang=np.linspace(0.03,0.96,100)
plt.plot(xrang,0*xrang, color='black', linewidth=2)
plt.xlim((0.03,0.96))
plt.ylim((-0.17,0.15))
plt.annotate('$x_s$', xy=(0.32,0.525),xycoords='figure fraction', fontsize=15)
plt.annotate('$x_0$', xy=(0.2,0.525),xycoords='figure fraction', fontsize=15)
plt.annotate('$x_0\'$', xy=(0.79,0.525),xycoords='figure fraction', fontsize=15)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.plot(0.2, 0,'o', color='black', mfc='white', markersize=8)


# The following function computes the whole system from just the rates and Hamiltonian
def simulate(Hfun,ratein,rateout,T,Nt,eps):
    equilibria = np.sort(solve(rateout-ratein,q))
    #semi-stable equilibria
    list1 = solve(rateout-ratein,q)
    list2 = solve(diff(rateout-ratein,q),q)
    list1_as_set = set(list1)
    intersection = list1_as_set.intersection(list2)
    semistable = list(intersection)
    difference = list1_as_set.difference(semistable)
    snsequilib = list(difference)
    N = len(snsequilib)
    #We want to mark stable points with 1, unstables with -1
    stability = np.empty((N,),int)
    stability[::2] = 1
    stability[1::2] = -1
    #Now we only need to determine stability for first equilibria, which we will determine by computing the sign
    eq = min(snsequilib)
    val = eq-eq/10 #this could be a problem is the function is not well defined at this point
    if((ratein-rateout).subs(q,val)<0):
        stability = -stability
    n_st=np.count_nonzero(stability==1)
    pval = np.zeros((Nt,2*n_st))
    qval = np.zeros((Nt,2*n_st))
    sval = np.zeros((Nt,2*n_st))
    j = 0
    for i in range(len(equilibria)):
        if (stability[i] == 1):
            q1,p1,s1 = modelsimfun(equilibria[i],Hfun,equilibria,ratein,rateout,T,Nt,eps,0.15,-0.2,1e-3)
            q2,p2,s2 = modelsimfun(equilibria[i],Hfun,equilibria,ratein,rateout,T,Nt,-eps,0.15,-0.2,1e-3)
            pval[:,j],pval[:,j+1] = p1,p2
            qval[:,j],qval[:,j+1] = q1,q2
            sval[:,j],sval[:,j+1] = s1,s2
            j = j+2
    offset = np.zeros(2*n_st)
    if (qval[-1,-1]<qval[-1,-2]):
        max1l = sval[-1,-2]#so that it cancels out (notice this is the right part)
    else:
        max1l = sval[-1,-1]
    for i in range(n_st):
        if (qval[-1,-1-2*i]<qval[-1,-2-2*i]):
            max2l = sval[-1,-1-2*i]
            max2r = sval[-1,-2-2*i]
            offset[-2*i-2], offset[-2*i-1] = max1l - max2r, max1l - max2r
            max1l = max2l + offset[-2*i-1]
        else:
            max2l = sval[-1,-2-2*i]
            max2r = sval[-1,-1-2*i]
            offset[-2*i-2], offset[-2*i-1] = max1l - max2r, max1l - max2r
            max1l = max2l + offset[-2*i-1]
    svaloffset = sval + offset 
    return pval,qval,sval,svaloffset


# Next we consider the second order WKB approximation, so we incorporate the prefactor:

def modelsimfunpref(x1,Hfun,equilibria,T,Nt,eps,uplimit,downlimit,eqeps):
    p,q = symbols('p q')
    Hfundp = lambdify([q,p],diff(Hfun,p))
    Hfundq = lambdify([q,p],diff(Hfun,q))
    Hfundp2 = lambdify([q,p],diff(diff(Hfun,p),p))
    Hfundq2 = lambdify([q,p],diff(diff(Hfun,q),q))
    Hfundqdp = lambdify([q,p],diff(diff(Hfun,q),p))
    eq=np.setdiff1d(equilibria,x1)
    def RHS(y,t):
        dy = np.zeros(5)
        dy[0] = Hfundp(y[0],y[1]) #q
        dy[1] = -Hfundq(y[0],y[1]) #p
        dy[2] = y[1]*dy[0]
        dy[3] = -(Hfundqdp(y[0],y[1])+1/2*y[4]*Hfundp2(y[0],y[1]))*y[3]
        dy[4] = -(Hfundp2(y[0],y[1])*y[4]**2+2*Hfundqdp(y[0],y[1])*y[4]+Hfundq2(y[0],y[1]))
        if dy[1] > 0 and y[1] > uplimit:
            y[1] = uplimit
            dy = np.zeros(5)
        if dy[1] < 0 and y[1] < downlimit:
            y[1] = downlimit
            dy = np.zeros(5)
        for i in range(len(eq)):
            if np.abs(y[0]-eq[i]) < eqeps:
                if y[1]<eqeps:#could probably get rid of this if
                    y[0]=eq[i]
                    y[1]=0
                    dy = np.zeros(5)
                    break
        return dy
    y0 = np.zeros(5)
    y0[0] = x1
    y0[1] = eps
    y0[2] = 0
    y0[3] = 1
    x = Symbol('x')   
    vals=solve(x**2*Hfundp2(x1,0)+2*x*Hfundqdp(x1,0),x)
    vals = vals[vals != 0]
    y0[4] =  vals    
    t = np.linspace(0,T,Nt)
    #compute solution
    #print("running simulation...")
    y = odeint(RHS,y0,t)
    q = y[:,0]
    p = y[:,1]
    s = y[:,2]
    k = y[:,3]
    z = y[:,4]
    #print("finished simulation")
    return(q,p,s,k,z)

# And we make a second function that computes the while system
# and also returns the matched prefactor

def simulatepref(Hfun,ratein,rateout,T,Nt,eps, uplimit,downlimit):
    equilibria2 = np.sort(solve(rateout-ratein,q))
    #semi-stable equilibria
    list1 = list(solveset(rateout-ratein,q).args[:])
    # list1 = solve(rateout-ratein,q)
    list2 = list(solveset(diff(rateout-ratein,q),q).args[:])
    # list2 = solve(diff(rateout-ratein,q),q)
    list1_as_set = set(list1)
    intersection = list1_as_set.intersection(list2)
    semistable = list(intersection)
    difference = list1_as_set.difference(semistable)
    snsequilib = list(difference)
    N = len(snsequilib)
    #We want to mark stable points with 1, unstables with -1
    stability = np.empty((N,),int)
    stability[::2] = 1
    stability[1::2] = -1
    #Now we only need to determine stability for first equilibria, which we will determine by computing the sign
    eq = min(snsequilib)
    val = eq-eq/10 #this could be a problem is the function is not well defined at this point
    if((ratein-rateout).subs(q,val)<0):
        stability = -stability
    n_st=np.count_nonzero(stability==1)
    pval = np.zeros((Nt,2*n_st))
    qval = np.zeros((Nt,2*n_st))
    sval = np.zeros((Nt,2*n_st))
    kval = np.zeros((Nt,2*n_st))
    zval = np.zeros((Nt,2*n_st))
    j = 0
    for i in range(len(equilibria)):
        if (stability[i] == 1):
            q1,p1,s1,k1,z1 = modelsimfunpref(equilibria[i],Hfun,equilibria,T,Nt,eps,uplimit,downlimit,1e-3)#0.4
            q2,p2,s2,k2,z2 = modelsimfunpref(equilibria[i],Hfun,equilibria,T,Nt,-eps,uplimit,downlimit,1e-3)
            pval[:,j],pval[:,j+1] = p1,p2
            qval[:,j],qval[:,j+1] = q1,q2
            sval[:,j],sval[:,j+1] = s1,s2
            kval[:,j],kval[:,j+1] = k1,k2
            zval[:,j],zval[:,j+1] = z1,z2
            j = j+2
    offset = np.zeros(2*n_st)
    if (qval[-1,-1]<qval[-1,-2]):
        max1l = sval[-1,-2]#so that it cancels out (notice this is the right part)
    else:
        max1l = sval[-1,-1]
    for i in range(n_st):
        if (qval[-1,-1-2*i]<qval[-1,-2-2*i]):
            max2l = sval[-1,-1-2*i]
            max2r = sval[-1,-2-2*i]
            offset[-2*i-2], offset[-2*i-1] = max1l - max2r, max1l - max2r
            max1l = max2l + offset[-2*i-1]
        else:
            max2l = sval[-1,-2-2*i]
            max2r = sval[-1,-1-2*i]
            offset[-2*i-2], offset[-2*i-1] = max1l - max2r, max1l - max2r
            max1l = max2l + offset[-2*i-1]
    svaloffset = sval + offset 

    offset = np.zeros(2*n_st)
    if (qval[-1,-1]<qval[-1,-2]):
        max1l = kval[-1,-2]#so that it cancels out (notice this is the right part)
    else:
        max1l = kval[-1,-1]
    for i in range(n_st):
        if (qval[-1,-1-2*i]<qval[-1,-2-2*i]):
            max2l = kval[-1,-1-2*i]
            max2r = kval[-1,-2-2*i]
            offset[-2*i-2], offset[-2*i-1] = max1l/max2r, max1l/max2r
            max1l = max2l*offset[-2*i-1]
        else:
            max2l = kval[-1,-2-2*i]
            max2r = kval[-1,-1-2*i]
            offset[-2*i-2], offset[-2*i-1] = max1l/max2r, max1l/max2r
            max1l = max2l*offset[-2*i-1]
    kvaloffset = kval*offset 
    return pval,qval,sval,svaloffset, kval, kvaloffset, zval
uplim = 0.3
downlim = -4.7
pval,qval,sval,svaloffset,kval,kvaloffset,zval =simulatepref(Hfun,ratein,rateout,500,10000,1e-12, uplim,downlim)
plt.plot(qval,kvaloffset)
plt.ylabel(r'$K(q)$')
plt.xlabel('Concentration')
plt.annotate('$x_0$', xy=(0.2,0.63),xycoords='figure fraction', fontsize=15)
plt.annotate('$x_s$', xy=(0.27,0.35),xycoords='figure fraction', fontsize=15)
plt.annotate('$x_0\'$', xy=(0.63,0.22),xycoords='figure fraction', fontsize=15)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.plot(equilibria, [36,11,1.2],'o', color='black', markersize=6)


# Next we would like to compare these results to simulations of the model
# To do this, we will use the Gillespie algorithm:

#Simulation with time
N=100
maxt=20000
#exponential rate is 1 as maximum (assuming w0+w1>1), so can take 3/2*maxt reactions to generate the uniforms
simulations=3000
startq=np.random.uniform(0,1,simulations)
uniforms=np.random.uniform(0,1,simulations*2*maxt)
uniformst=np.random.uniform(0,1,simulations*2*maxt)
loguniformst=np.log(uniformst)
concentration = np.zeros(simulations)

for j in range(simulations):
    nx=0.4*N
    i=0
    t=0
    while t<maxt:
        qx=nx/N
        wplus=qx**2+a
        wminus=qx**3+K*qx
        w=wminus+wplus#ratein.subs(q,qx)/(ratein.subs(q,qx)+rateout.subs(q,qx))
        dt=-loguniformst[maxt*j+i]/w
        t=t+dt
        if wplus/w>uniforms[maxt*j+i] and qx<1:
            nx=nx+1
        else:
            nx=nx-1
        i+=1
    concentration[j]=nx/N


data= pd.DataFrame([qval.flatten('F'),svaloffset.flatten('F'),kvaloffset.flatten('F')])
sorteddata=data.sort_values(by=0, axis=1)
area1=sum([np.abs(scp.integrate.trapz(np.exp(-N*svaloffset[:,i]),qval[:,i])) for i in range(4)])
area2=sum([np.abs(scp.integrate.trapz(kvaloffset[:,i]*np.exp(-N*svaloffset[:,i]),qval[:,i])) for i in range(4)])
plt.plot(sorteddata.iloc[0,:],1/(area1)*np.exp(-N*sorteddata.iloc[1,:]),label='No prefactor')
plt.plot(sorteddata.iloc[0,:],1/(area2)*sorteddata.iloc[2,:]*np.exp(-N*sorteddata.iloc[1,:]),label='Prefactor')
plt.hist(concentration, bins=101,density=True,histtype='stepfilled',alpha=0.5,facecolor='skyblue',
         edgecolor='steelblue')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()


uplim = 0.3
downlim = -1.7
T=500
Nt=1400
pval,qval,sval,svaloffset, kval, kvaloffset, zval = simulatepref(Hfun,ratein,rateout,T,Nt,1e-12, uplim, downlim)
Hfundp2=lambdify([q,p],diff(diff(Hfun,p),p))
Hfundpdq=lambdify([q,p],diff(diff(Hfun,p),q))
Hfundpdq2 = lambdify([q,p],diff(diff(diff(Hfun,p),q),q))


# First we will look at the relaxation
rate=[q**2+a,q**3+K*q]
react=[1,-1]
equ=solve((Matrix(rate).T*Matrix(react)),q)
qs=np.array(equ[1][0],dtype=complex).real
ls = np.array(np.sqrt(Hfundp2(qs,0)/(N*Hfundpdq(qs,0))),dtype = 'complex').real

def pi_ls(q,lsfac,N):
    val = scp.special.erfc((q-qs)/lsfac)
    return N*lsfac/Hfundp2(qs,0)*exp((q-qs)**2/lsfac**2)*np.sqrt(np.pi)*val
n_ls = 300
x = np.linspace(0,1,n_ls)
xr = np.linspace(0,1,300)
pifunct = np.zeros(n_ls)
for i in range(n_ls):
    pifunct[i] = pi_ls(x[i], ls, N)

plt.plot(xr,1/Hfundp(xr,0),label='Relaxation')
plt.plot(x,pifunct, label='Approximation')
plt.ylim((20,1000))
plt.xlim((0,1))
plt.annotate('$x_s$', xy=(0.28,0.17),xycoords='figure fraction', fontsize=13)
plt.scatter(0.2,20, marker='|', s=100, color='r').set_clip_on(False)
plt.annotate('$x_0$', xy=(0.17,0.17),xycoords='figure fraction', fontsize=13)
plt.scatter(equ[2][0],20, marker='|', s=100, color='r').set_clip_on(False)
plt.annotate('$x_0\'$', xy=(0.74,0.17),xycoords='figure fraction', fontsize=13)
plt.scatter(equ[0][0],20, marker='|', s=100, color='r').set_clip_on(False)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()
plt.xlabel('Concentration')


# Match for the activation

n_ls = np.shape(qval[:,0])[0]
xr = np.linspace(0.29,0.73,300)
x = np.linspace(0,0.2,n_ls)
pifunct1 = np.zeros(n_ls)
for i in range(n_ls):
    pifunct1[i] = pi_ls(qval[i,0], ls, N)

pifunct2 = np.zeros(n_ls)
for i in range(n_ls):
    pifunct2[i] = pi_ls(qval[i,1], ls, N)


funct1 = 2*N*ls*np.sqrt(np.pi)/Hfundp2(0.2,0)*np.exp((qval[:,0]-0.2)**2/ls**2)
funct2 = 2*N*ls*np.sqrt(np.pi)/Hfundp2(0.2,0)*np.exp((qval[:,1]-0.2)**2/ls**2)

prop = (np.exp(-100*sval[-1,0])*kval[-1,0])/funct1[-1]


plt.plot(qval[:,1],np.exp(-100*sval[:,1])*kval[:,1]+prop*pifunct2-prop*funct2, label='Matched Activation', color = 'b')
plt.plot(qval[:,0],np.exp(-100*sval[:,0])*kval[:,0]+prop*pifunct1-prop*funct1, color = 'b')
plt.plot(qval[:,1],np.exp(-100*sval[:,1])*kval[:,1],label='Activation', color = 'gray')
plt.plot(qval[:,0],np.exp(-100*sval[:,0])*kval[:,0], color = 'gray')
plt.plot(qval[:,1],prop*pifunct2, label='Approximation', color='g')
plt.plot(qval[:,0],prop*pifunct1, color='g')
plt.plot(qval[:,1],prop*funct2, label='Asymptotic Approximation', color='r')
plt.plot(qval[:,0],prop*funct1, color='r')
plt.ylim((0,1.2))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()
plt.xlabel('Concentration')

# Now the relaxation trajectory
n_ls1 = 1000
N=100
xr = np.linspace(0.2,0.73,n_ls1)
pifunct = np.zeros(n_ls1)
for i in range(n_ls1):
    pifunct[i] = pi_ls(xr[i], ls, N) #app
# Here we need to go one order smaller in the approximation by considering the next derivative:
funct = 1/(Hfundpdq(0.2,0)*(xr-0.2)+1/2*Hfundpdq2(0.2,0)*(xr-0.2)**2) #asympt

plt.plot(xr[1:], (1/Hfundp(xr,0)+pifunct-funct)[1:] , label='Matched relaxation')
plt.plot(xr[1:], 1/Hfundp(xr,0)[1:] , label='Relaxation')
plt.plot(xr, funct , label='Asymptotic Approximation')
plt.plot(xr, pifunct , label='Approximation')
plt.yscale('log')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()
plt.xlabel('Concentration')

# And finally we match both trajectories using the same proportionality constant as before
plt.plot(qval[:,1],np.exp(-100*sval[:,1])*kval[:,1]+prop*pifunct2-prop*funct2, color = 'royalblue')
plt.plot(qval[:,0],np.exp(-100*sval[:,0])*kval[:,0]+prop*pifunct1-prop*funct1, color = 'royalblue')
plt.plot(xr, prop*(1/Hfundp(xr,0)+pifunct-funct), color='royalblue')
plt.ylim((0,1.2))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Concentration')
plt.ylabel('Probability')


# SIR model

# We will now move on to the SIR model
# We can define this model in the following way

beta=1.3
eta=1e-6
gamma=1/13
mu=5.5e-5
xvar='x1 x2'
pvar='p1 p2'
xs=symbols(xvar)
ps=symbols(pvar)
rate=[beta*xs[0]*xs[1]+eta*xs[0],gamma*xs[1],mu*xs[1],mu*(1-xs[0]-xs[1])]
react=[[-1,1],[0,-1],[1,-1],[1,0]]
Hfun = 0
S=len(rate)
n=len(ps)
# define the Hamiltonian
for i in range(S):
    Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun
Hfun
equ = solve((-rate[0]+rate[2]+rate[3],rate[0]-rate[1]-rate[2]),xs)
# The unique fixed point
x1 = equ[0]


# To do this, we will have to change our function above to make it work with more dimensions.
# We will start considering the prefactor from the beginning

def trajectorypref(x1,Hfun,n,T,Nt,pertbt,maximum):
    Hfundp = lambdify([xs,ps],derive_by_array(Hfun,ps))
    Hfundx = lambdify([xs,ps],derive_by_array(Hfun,xs))
    Hfundp2 = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,ps),ps))
    Hfundx2 = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),xs))
    Hfundxdp = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),ps))
    Hfundpdx = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,ps),xs))
    
    def RHS1(y,t):
        dy = np.zeros(2*n+1)
        subst1 = [y[i] for i in range(n)]#xs
        subst2 = [y[n+i] for i in range(n)]#ps
        dy[0:n] = Hfundp(subst1,subst2) #q
        dy[n:2*n] = -np.array(Hfundx(subst1,subst2)) #p
        dy[2*n] = y[n:2*n].dot(dy[0:n]) #s
        return dy
    def RHS2(y,t):
        dy = np.zeros(1+n**2)
        subst1 = [xfunc(t),yfunc(t)]#xs
        subst2 = [p1func(t), p2func(t)]#ps
        Hfundp2mat = Hfundp2(subst1,subst2)
        Hfundx2mat = Hfundx2(subst1,subst2)
        Hfundxdpmat = Hfundxdp(subst1,subst2)
        Hfundpdxmat = Hfundpdx(subst1,subst2)
        z = Matrix(np.reshape(y[1:],(n,n), order='F'))
        dy[0] = -(np.trace(np.array(Hfundpdxmat))+1/2*np.sum(np.multiply(np.array(z),np.array(Hfundp2mat))))*y[0]
        zdot = -(np.matmul(np.matmul(np.array(z, dtype=float),np.array(Hfundp2mat)),np.array(z, dtype=float))+ np.matmul(np.array(Hfundxdpmat),np.array(z, dtype=float))+ np.matmul(np.array(z, dtype=float),np.array(Hfundpdxmat))+np.array(Hfundx2mat)) 
        dy[1:] = np.reshape(zdot,n**2,order='F')
        if (y[0]>maximum) & (t>1000):
            dy[0] = 0
        return dy
    y0 = np.zeros(2*n+2+n**2)
    y0[0:n] = x1+pertbt #x0
    y0[2*n] = 0 #s
    subst1 = [x1[i] for i in range(n)]
    subst2 = [0 for i in range(n)]
    B = Hfundp2(subst1,subst2)
    A = Hfundxdp(subst1,subst2)
    inversemat = np.linalg.inv(sslin.solve_continuous_lyapunov(np.array(A,dtype='float'),-np.array(B,dtype='float')))
    y0[2*n+2:] = np.reshape(inversemat,4,'F') #z
    y0[n:2*n] =  inversemat.dot(pertbt) #p
    y0[2*n+1] = 2/np.pi*np.linalg.det(inversemat) # k/ missing N value so that it does not grow too much
    t = np.linspace(0,T,Nt)
    #compute solution
    y1 = odeint(RHS1,y0[:2*n+1],t)
    xfunc = interp1d(t, y1[:,0], bounds_error=False, fill_value="extrapolate")
    yfunc = interp1d(t, y1[:,1], bounds_error=False, fill_value="extrapolate")
    p1func = interp1d(t, y1[:,2], bounds_error=False, fill_value="extrapolate")
    p2func = interp1d(t, y1[:,3], bounds_error=False, fill_value="extrapolate")
    y2 = odeint(RHS2,y0[2*n+1:],t,hmax=1)
    return(y1,y2)
# Some code to generate a random perturbation:
def perturbcircle(n,nperturb,eps):
    randomn = np.random.random(nperturb)*2*np.pi
    vectorarray = []
    for i in range(nperturb):
        vectorarray.append(np.array([eps*np.cos(randomn[i]),eps*np.sin(randomn[i])]))
    return(vectorarray)

# Let's make a few plots to see what the trajectories look like
x1=equ[0]
n=2
Nt=70000
eps=1e-9
T=34000#31150 #21898
pertbt = perturbcircle(2,1,eps)[0]
y1,y2=trajectorypref(x1,Hfun,2,T, Nt, pertbt,4000)
xvec=y1[:,:2]
pvec=y1[:,2:4]
svec=y1[:,4]
kvec=y2[:,0]
zvec=y2[:,1:]
plt.plot(kvec[:40000])

k=46890
ax = plt.axes(projection='3d')
ax.plot3D(y1[:k,0], y1[:k,1], y2[:k,0])

plt.plot(y2[:46500,1])
plt.plot(y2[:46500,2])
plt.plot(y2[:46500,3])
np.min(y2[:900,1]**2)
plt.plot(xvec[:,0],xvec[:,1])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


plt.plot(pvec[:,0],pvec[:,1])
plt.xlabel(r"$P_x$")
plt.ylabel(r"$P_y$")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


plt.plot(xvec[:,0],xvec[:,1])
plt.plot(svec)
plt.title("S")
plt.ylabel("S")

plt.plot(kvec)
plt.title("Prefactor")
plt.ylabel("K")

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('S')
ax.plot3D(xvec[:,0], xvec[:,1], svec)

ax = plt.axes(projection='3d')
ax.plot3D(xvec[:,0], xvec[:,1], kvec)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('K')
#without prefactor
ax = plt.axes(projection='3d')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$P(x,y)$')
k = 68000
N=6000000
ax.plot3D(xvec[:k,0], xvec[:k,1], kvec[:k]*np.exp(-N*svec[:k]),color='orangered', linestyle='-')
N=900000
ax.plot3D(xvec[:k,0], xvec[:k,1], kvec[:k]*np.exp(-N*svec[:k]),color='green', linestyle='-.')
N=300000
ax.plot3D(xvec[:k,0], xvec[:k,1], kvec[:k]*np.exp(-N*svec[:k]),color='purple', linestyle='--')
N=50000
ax.plot3D(xvec[:k,0], xvec[:k,1], kvec[:k]*np.exp(-N*svec[:k]),color='darkturquoise', linestyle=(0,(1,1)))


ax.set_title('N=%i' %N)
#with prefactor
ax = plt.axes(projection='3d')
N=10000000
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('P(x,y) not scaled')
ax.plot3D(xvec[:,0], xvec[:,1], kvec*np.exp(-N*svec))
ax.set_title('N=%i' %N)

def perturb(n,nperturb,eps):
    vectmat = eps*np.identity(n)
    randomn = np.random.random(2*nperturb)*2-1
    vectorarray = []
    for i in range(nperturb):
        vectorarray.append(vectmat[:,0]*randomn[i]+vectmat[:,1]*randomn[nperturb+i])
    return(vectorarray)

def modelsimfunprefn2(x1,Hfun,equilibria,n,T,Nt,pertbt):
    Hfundp = derive_by_array(Hfun,ps)
    Hfundx = derive_by_array(Hfun,xs)
    Hfundp2 = derive_by_array(derive_by_array(Hfun,ps),ps)
    Hfundx2 = derive_by_array(derive_by_array(Hfun,xs),xs)
    Hfundxdp = derive_by_array(derive_by_array(Hfun,xs),ps)
    Hfundpdx = derive_by_array(derive_by_array(Hfun,ps),xs)
    eq = np.setdiff1d(equilibria,x1)
    N = 1000 #later we will add this to definition
    def RHS(y,t):
        dynew = np.zeros(2*n+2+n**2)
        subst1 = [(xs[i],y[i]) for i in range(n)]
        subst2 = [(ps[i],y[n+i]) for i in range(n)]
        Hfundp2mat = Hfundp2.subs(subst2+subst1).tomatrix()
        Hfundx2mat = Hfundx2.subs(subst2+subst1).tomatrix()
        Hfundxdpmat = Hfundxdp.subs(subst2+subst1).tomatrix()
        Hfundpdxmat = Hfundpdx.subs(subst2+subst1).tomatrix()
        dynew[0:n] = Hfundp.subs(subst1+subst2) #q
        dynew[n:2*n] = -Hfundx.subs(subst2+subst1) #p
        dynew[2*n] = y[n:2*n].dot(dynew[0:n])
        z = Matrix(np.reshape(y[2*n+2::],(n,n), order='F'))
        dynew[2*n+1] = -(trace(Hfundpdxmat)+1/2*sum(z.multiply_elementwise(Hfundp2mat)))*y[2*n+1]
        zdot = -(z.multiply(Hfundp2mat).multiply(z)+ Hfundxdpmat.multiply(z)+ z.multiply(Hfundpdxmat)+Hfundx2mat) 
        dynew[2*n+2:] = np.reshape(zdot,n**2,order='F')
        if sum(abs(y))<10000:
            dy = dynew
        else:
            dy = np.zeros(2*n+2+n**2)
        return dy
    y0 = np.zeros(2*n+2+n**2)
    y0[0:n] = x1+pertbt #x0
    y0[2*n] = 0 #s
    subst1 = [(xs[i],x1[i]) for i in range(n)]
    subst2 = [(ps[i],0) for i in range(n)]
    B = Hfundp2.subs(subst2+subst1).tomatrix()
    A = Hfundxdp.subs(subst2+subst1).tomatrix()
    inversemat = np.linalg.inv(sslin.solve_continuous_lyapunov(np.array(A,dtype='float'),np.array(-B,dtype='float')))
    y0[2*n+2:] = np.reshape(inversemat,4,'F') #z
    y0[n:2*n] = inversemat.dot(pertbt) #p
    y0[2*n+1] = 1 #k
    t = np.linspace(0,T,Nt)
    #compute solution
    y = odeint(RHS,y0,t)
    return(y)

def perturbcircle(n,nperturb,eps):
    randomn = np.random.random(nperturb)*2*np.pi
    vectorarray = []
    for i in range(nperturb):
        vectorarray.append(np.array([eps*np.cos(randomn[i]),eps*np.sin(randomn[i])]))
    return(vectorarray)

def modelperturb3circle(x1,Hfun,n,T,Nt,nperturb,eps,maximum):
    pertvect = perturbcircle(n,nperturb,eps)
    data = pd.DataFrame()
    for i in range(nperturb):
        y1,y2=trajectorypref(x1,Hfun,n,T,Nt,pertvect[i],maximum)
        y=np.append(y1,y2,axis=1)
        if np.max(y[:,2*n+1])<maximum:
            dat1 = pd.DataFrame(y)
            data = data.append(dat1)
        else:
            index = np.where(y[:,2*n+1]>maximum)[0][0]
            dat1 = pd.DataFrame(y[:index,:])
            data = data.append(dat1)
    return(data)

def probint(prob,yval):
    n = len(prob)
    if n == 0:
        return(0)
    if n == 1:
        return(0)
    dat = pd.DataFrame(np.array((yval,prob)).T,columns=['y_values','probabilities'])
    datsorted = dat.sort_values(by=['y_values'])
    appint = 0
    i = 0
    for i in range(n-1):
        appint = appint+(datsorted.iloc[i+1,0]-datsorted.iloc[i,0])*datsorted.iloc[i,1]
    return(appint)

def bin_probp(data,x,n,width,N,normalc):
    #n first columns of data are spatial coordinates, so want 
    #to keep only the datapoint within the width of the point
    newdata = data[(data.iloc[:,0]<x+width) & (data.iloc[:,0]>x-width)]
    yval = newdata.iloc[:,1]
    xval = x
    prob = 1/normalc*newdata.iloc[:,5]*np.exp(-N*newdata.iloc[:,4])
    probinteg = probint(prob,yval)
    return(probinteg)

def bin_probavnp(data,x,n,width,N,normalc):
    #n first columns of data are spatial coordinates, so want 
    #to keep only the datapoint within the width of the point
    newdata = data[(data.iloc[:,0]<x+width) & (data.iloc[:,0]>x-width)]
    if newdata.size==0:
        return(0)
    yval = newdata.iloc[:,1]
    xval = x
    prob = 1/normalc*newdata.iloc[:,5]*np.exp(-N*newdata.iloc[:,4])
    probav = np.mean(prob)
    return(probav)

def margdistbinavnp(data,nx,N,n,normalc=1):
    yvec = data.iloc[:,1]
    xvec = data.iloc[:,0]
    xpt = np.linspace(np.min(xvec),np.max(xvec),nx)
    width = abs(np.min(xvec)-np.max(xvec))/(2*(nx-1))
    xpt = np.delete(np.delete(xpt,0),-1)
    probaav = np.zeros(nx-2)
    l = 0
    for i in xpt:
        probaav[l] = bin_probavnp(data,i,n,width,N, normalc)
        l = l+1
    return(xpt,probaav)

dataset1 = modelperturb3circle(x1,Hfun,n,T,Nt,10,1e-9,4000)

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('P(x,y) not scaled')
N=3000000
ax.plot3D(dataset1[0], dataset1[1], dataset1[5]*np.exp(-N*dataset1[4]))

nx = 1000
N=3000000
xpt,probainteg = margdistbinavnp(dataset1,nx,N,n)

plt.plot(xpt,probainteg,'.')
plt.xlabel(r"$x$")
plt.ylabel(r"$P(x)$")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


prob=dataset1[5]*np.exp(-N*dataset1[4])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(dataset1[0], dataset1[1], prob, cmap=cm.jet, linewidth=0.1, antialiased=False)
fig.colorbar(surf)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability')


# Toggle-switch model

# We will now move on to study the Toggle-switch model

# We will look at the first model that we looked at, defined as

alpha=10 
gamma=2
mu=0.5
xvar='x1 x2'
pvar='p1 p2'
xs=symbols(xvar)
ps=symbols(pvar)
rate=[alpha/(1+(xs[0])**gamma),mu*xs[1],alpha/(1+(xs[1])**gamma),mu*xs[0]]
react=[[0,1],[0,-1],[1,0],[-1,0]]
Hfun = 0
N=len(rate)
n=len(ps)
# define the Hamiltonian
for i in range(N):
    Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun
Hfun
#Next we would like to find the equilibrium point
equ=np.array(solve([rate[2]-rate[3],rate[0]-rate[1]],xs)).astype(np.complex).real


# We need to find the perturbation that makes the switch
# to do this, we define a shooting problem from the stable point
# to the unstable one and we  minimise this wrt the angle and with
# a fixed radius. In this way, we consider all trajectories

# The first distance we might think of is the min Euclidean distance between the trajectory and the unstable point
 
def trajectory2euc(ang_theta,x1,x2,Hfun,n,T,Nt,r,xlim,ylim,return_distance=False,complete=True, trajectory=False):
    pertbt = np.array([cmath.rect(r,ang_theta).real,cmath.rect(r,ang_theta).imag])
    Hfundp = lambdify([xs,ps],derive_by_array(Hfun,ps))
    Hfundx = lambdify([xs,ps],derive_by_array(Hfun,xs))
    Hfundp2 = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,ps),ps))
    Hfundx2 = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),xs))
    Hfundxdp = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),ps))
    Hfundpdx = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,ps),xs))
    
    def RHS1(y,t):
        dy = np.zeros(2*n+1)
        subst1 = [y[i] for i in range(n)]#xs
        subst2 = [y[n+i] for i in range(n)]#ps
        dy[0:n] = Hfundp(subst1,subst2) #q
        dy[n:2*n] = -np.array(Hfundx(subst1,subst2)) #p
        dy[2*n] = y[n:2*n].dot(dy[0:n]) #s
        if (y[0]>xlim) | (y[1]>ylim):
            dy = np.zeros(2*n+1)
        return dy
    def RHS2(y,t):
        dy = np.zeros(1+n**2)
        subst1 = [xfunc(t),yfunc(t)]#xs
        subst2 = [p1func(t), p2func(t)]#ps
        Hfundp2mat = Hfundp2(subst1,subst2)
        Hfundx2mat = Hfundx2(subst1,subst2)
        Hfundxdpmat = Hfundxdp(subst1,subst2)
        Hfundpdxmat = Hfundpdx(subst1,subst2)
        z = Matrix(np.reshape(y[1:],(n,n), order='F'))
        dy[0] = -(np.trace(np.array(Hfundpdxmat))+1/2*np.sum(np.multiply(np.array(z),np.array(Hfundp2mat))))*y[0]
        zdot = -(np.matmul(np.matmul(np.array(z, dtype=float),np.array(Hfundp2mat)),np.array(z, dtype=float))+ np.matmul(np.array(Hfundxdpmat),np.array(z, dtype=float))+ np.matmul(np.array(z, dtype=float),np.array(Hfundpdxmat))+np.array(Hfundx2mat)) 
        dy[1:] = np.reshape(zdot,n**2,order='F')
        for j in range(n**2):
            if (abs(y[1+j])<0.02) & (t>5):
                dy[1+j]=0
                dy[0]=0
        return dy
    y0 = np.zeros(2*n+2+n**2)
    y0[0:n] = x1+pertbt #x0
    y0[2*n] = 0 #s
    subst1 = [x1[i] for i in range(n)]
    subst2 = [0 for i in range(n)]
    B = Hfundp2(subst1,subst2)
    A = Hfundxdp(subst1,subst2)
    inversemat = np.linalg.inv(sslin.solve_continuous_lyapunov(np.array(A,dtype='float'),-np.array(B,dtype='float')))
    y0[2*n+2:] = np.reshape(inversemat,4,'F') #z
    y0[n:2*n] =  inversemat.dot(pertbt) #p
    y0[2*n+1] = 2*N/np.pi*np.linalg.det(inversemat) #1 #k
    t = np.linspace(0,T,Nt)
    #compute solution
    y1 = odeint(RHS1,y0[:2*n+1],t)
    index = np.argmin(np.sqrt((y1[:,0]-x2[0])**2+(y1[:,1]-x2[1])**2))
    if return_distance==True:
        y1[(y1[:,0]==0)&(y1[:,1]==0),:2] = xlim 
        distance = np.min(np.sqrt((y1[:,0]-x2[0])**2+(y1[:,1]-x2[1])**2))
        return(distance)
    if trajectory==True:
        return(y1)
    xfunc = interp1d(t, y1[:,0], bounds_error=False, fill_value="extrapolate")
    yfunc = interp1d(t, y1[:,1], bounds_error=False, fill_value="extrapolate")
    p1func = interp1d(t, y1[:,2], bounds_error=False, fill_value="extrapolate")
    p2func = interp1d(t, y1[:,3], bounds_error=False, fill_value="extrapolate")
    y2 = odeint(RHS2,y0[2*n+1:],t,hmax=1)
    if complete:
        return(y1,y2)
    else:
        return(y1[:index,:],y2[:index,:])

# We can do a plot of how this distance looks 
Nt=60000
T=60
l=1e-7

distanceval=np.zeros(2000)
rangetheta=np.linspace(0,2*np.pi,2000)
for i in range(2000):
    distanceval[i]=trajectory2euc(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distanceval)

distanceval1=np.zeros(200)
rangetheta=np.linspace(4.5,5,200)
for i in range(200):
    distanceval1[i]=trajectory2euc(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distanceval1)

distanceval2=np.zeros(200)
rangetheta=np.linspace(4.7,4.8,200)
for i in range(200):
    distanceval2[i]=trajectory2euc(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distanceval2)

distanceval3=np.zeros(2000)
rangetheta=np.linspace(4.758,4.766,2000)
for i in range(2000):
    distanceval3[i]=trajectory2euc(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distanceval3)
plt.xlabel(r"$x$")
plt.ylabel("Distance")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

# We can see that the Euclidean norm is not an appropiate distance here
# we will see why later, but now we will change the definition of the distance 

# We tried over 10 distances, but this one is the one that worked best
def trajectory2(ang_theta,x1,x2,Hfun,n,T,Nt,r,xlim,ylim,return_distance=False,complete=True, trajectory=False):
    pertbt = np.array([cmath.rect(r,ang_theta).real,cmath.rect(r,ang_theta).imag])
    Hfundp = lambdify([xs,ps],derive_by_array(Hfun,ps))
    Hfundx = lambdify([xs,ps],derive_by_array(Hfun,xs))
    Hfundp2 = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,ps),ps))
    Hfundx2 = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),xs))
    Hfundxdp = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),ps))
    Hfundpdx = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,ps),xs))
    
    def RHS1(y,t):
        dy = np.zeros(2*n+1)
        subst1 = [y[i] for i in range(n)]#xs
        subst2 = [y[n+i] for i in range(n)]#ps
        dy[0:n] = Hfundp(subst1,subst2) #q
        dy[n:2*n] = -np.array(Hfundx(subst1,subst2)) #p
        dy[2*n] = y[n:2*n].dot(dy[0:n]) #s
        if (y[0]>xlim) | (y[1]>ylim):
            dy = np.zeros(2*n+1)
        return dy
    def RHS2(y,t):
        dy = np.zeros(1+n**2)
        subst1 = [xfunc(t),yfunc(t)]#xs
        subst2 = [p1func(t), p2func(t)]#ps
        Hfundp2mat = Hfundp2(subst1,subst2)
        Hfundx2mat = Hfundx2(subst1,subst2)
        Hfundxdpmat = Hfundxdp(subst1,subst2)
        Hfundpdxmat = Hfundpdx(subst1,subst2)
        z = Matrix(np.reshape(y[1:],(n,n), order='F'))
        dy[0] = -(np.trace(np.array(Hfundpdxmat))+1/2*np.sum(np.multiply(np.array(z),np.array(Hfundp2mat))))*y[0]
        zdot = -(np.matmul(np.matmul(np.array(z, dtype=float),np.array(Hfundp2mat)),np.array(z, dtype=float))+ np.matmul(np.array(Hfundxdpmat),np.array(z, dtype=float))+ np.matmul(np.array(z, dtype=float),np.array(Hfundpdxmat))+np.array(Hfundx2mat)) 
        dy[1:] = np.reshape(zdot,n**2,order='F')
        for j in range(n**2):
            if (abs(y[1+j])<0.02) & (t>5):
                dy[1+j]=0
                dy[0]=0
    
        return dy
    y0 = np.zeros(2*n+2+n**2)
    y0[0:n] = x1+pertbt #x0
    y0[2*n] = 0 #s
    subst1 = [x1[i] for i in range(n)]
    subst2 = [0 for i in range(n)]
    B = Hfundp2(subst1,subst2)
    A = Hfundxdp(subst1,subst2)
    inversemat = np.linalg.inv(sslin.solve_continuous_lyapunov(np.array(A,dtype='float'),-np.array(B,dtype='float')))
    y0[2*n+2:] = np.reshape(inversemat,4,'F') #z
    y0[n:2*n] =  inversemat.dot(pertbt) #p
    y0[2*n+1] = 2*N/(np.pi)*np.linalg.det(inversemat) #1 #k
    t = np.linspace(0,T,Nt)
    #compute solution
    y1 = odeint(RHS1,y0[:2*n+1],t)#, full_output=True)
    index = np.argmin(np.sqrt(((y1[:,0]-x2[0])/(x1[0]-x2[0]))**2+((y1[:,1]-x2[1])/(x1[1]-x2[1]))**2))
    if return_distance==True:
        y1[(y1[:,0]==0)&(y1[:,1]==0),:2] = xlim 
        distance = np.min(np.sqrt(((y1[:,0]-x2[0])/(x1[0]-x2[0]))**2+((y1[:,1]-x2[1])/(x1[1]-x2[1]))**2))
        return(distance)
    if trajectory==True:
        return(y1)
    xfunc = interp1d(t, y1[:,0], bounds_error=False, fill_value="extrapolate")
    yfunc = interp1d(t, y1[:,1], bounds_error=False, fill_value="extrapolate")
    p1func = interp1d(t, y1[:,2], bounds_error=False, fill_value="extrapolate")
    p2func = interp1d(t, y1[:,3], bounds_error=False, fill_value="extrapolate")
    y2 = odeint(RHS2,y0[2*n+1:],t,hmax=1)
    if complete:
        return(y1,y2)
    else:
        return(y1[:index,:],y2[:index,:])

# We now repeat the plots before to see how they change with the new distance 

Nt=60000
T=60
l=1e-7

distance2val=np.zeros(2000)
rangetheta=np.linspace(0,2*np.pi,2000)
for i in range(2000):
    distance2val[i]=trajectory2(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distance2val)

distance2val1=np.zeros(200)
rangetheta=np.linspace(4.5,5,200)
for i in range(200):
    distance2val1[i]=trajectory2(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distanceval1)

distance2val2=np.zeros(200)
rangetheta=np.linspace(4.7,4.8,200)
for i in range(200):
    distance2val2[i]=trajectory2(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distanceval2)

distance2val3=np.zeros(2000)
rangetheta=np.linspace(4.758,4.766,2000)
for i in range(2000):
    distance2val3[i]=trajectory2(rangetheta[i],equ[0],equ[1],Hfun,n,T,Nt,l,40,40, return_distance=True)
plt.plot(rangetheta,distance2val3)
plt.xlabel(r"$x$")
plt.ylabel("Distance")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


# We will now plot the switching path and the trajectories close to it

result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,True),tol=1e-3,atol=1e-5)
angle2=result2['x']


# We will now visualise trajectories
path1,b=trajectory2(angle2,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False)
plt.plot(path1[:,0],path1[:,1], label='1')

path1=trajectory2(4.762,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False,trajectory=True)
plt.plot(path1[:30370,0],path1[:30370,1], label='1')

path1=trajectory2(angle2-1e-8,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False,trajectory=True)
plt.plot(path1[:41370,0],path1[:41370,1], label='1')

path2=trajectory2(angle2-1e-7,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False,trajectory=True)
plt.plot(path2[:39660,0],path2[:39660,1], label='2')

path3=trajectory2(angle2-1e-6,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False,trajectory=True)
plt.plot(path3[:37950,0],path3[:37950,1], label='3')

path4=trajectory2(angle2-1e-5,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False,trajectory=True)
plt.plot(path4[:36150,0],path4[:36150,1], label='4')

path4=trajectory2(angle2-1e-4,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False,trajectory=True)
plt.plot(path4[:34200,0],path4[:34200,1], label='4')

path4=trajectory2(angle2-1e-3,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False,trajectory=True)
plt.plot(path4[:32100,0],path4[:32100,1], label='4')

path4,b=trajectory2(angle2+1e-7,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False)
plt.plot(path4[:,0],path4[:,1], label='4')

path4,b=trajectory2(angle2+1e-5,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False)
plt.plot(path4[:,0],path4[:,1], label='4')

path4,b=trajectory2(angle2+1e-4,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False)
plt.plot(path4[:,0],path4[:,1], label='4')

path4,b=trajectory2(angle2+1e-3,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False)
plt.plot(path4[:,0],path4[:,1], label='4')

path4,b=trajectory2(angle2+1e-2,equ[0],equ[1],Hfun,n,T,Nt,1e-5,40,40,complete=False)
plt.plot(path4[:,0],path4[:,1], label='4')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.annotate('$x_0$', xy=(0.18,0.8941),xycoords='figure fraction', fontsize=13)
plt.plot(equ[1][0],equ[1][1],'o', color='black', mfc='white',markersize=6)
plt.annotate('$x_s$', xy=(0.88,0.17),xycoords='figure fraction', fontsize=13)
plt.plot(equ[0][0],equ[0][1], 'o', color='black', markersize=6)


# We will begin with the model without self-feedback, so a_i=0
a1=0
a2=0
b1=1
b2=1
omega=0.5
k1=1
k2=1
m=4

xvar='x1 x2'
pvar='p1 p2'
xs=symbols(xvar)
ps=symbols(pvar)
rate=[a1*xs[0]**m/(omega**m+xs[0]**m)+ b1*omega**m/(omega**m+(xs[1])**m),k1*xs[0],a2*xs[1]**m/(omega**m+xs[1]**m)+ b2*omega**m/(omega**m+(xs[0])**m),k2*xs[1]]
react=[[1,0],[-1,0],[0,1],[0,-1]]
Hfun = 0
N=len(rate)
n=len(ps)
# define the Hamiltonian
for i in range(N):
    Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun


equ1=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.5,0.5))).astype(np.complex).real
equ2=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(b1,0.01))).astype(np.complex).real
equ3=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.01,b2))).astype(np.complex).real

equ1=np.array([equ1[0][0],equ1[1][0]])
equ2=np.array([equ2[0][0],equ2[1][0]])
equ3=np.array([equ3[0][0],equ3[1][0]])

# Plot vector field
x,y = np.meshgrid(np.linspace(0,1.2,15),np.linspace(0,1.2,10))
u = b1*omega**m/(omega**m+y**m)-k1*x
v = b2*omega**m/(omega**m+x**m)-k2*y

fig, ax = plt.subplots()

ax.streamplot(x,y,u,v, density = 1.7, color='grey')
plt.plot(equ1[0],equ1[1],'o', color='black', mfc='white', markersize=8)
plt.plot(equ2[0],equ2[1],'o', color='black', markersize=8)
plt.plot(equ3[0],equ3[1],'o', color='black', markersize=8)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.ylim((0,1.2))
plt.xlim((0,1.2))
ax.text(
    0.06, 1.1, "$x_0$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
ax.text(
    0.5, 0.6, "$x_s$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

ax.text(
    1, 0.18, "$x_0\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))


# next we would like to look at the switching trajectories.

result1=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,n,T,Nt,1e-7,40,40,True),tol=1e-3,atol=1e-5)
angle1=result1['x']
switch1,y2=trajectory2(angle1,equ3,equ1,Hfun,n,T,Nt,1e-7,40,40,complete=False)


result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ2,equ1,Hfun,n,T,Nt,1e-7,40,40,True),tol=1e-3,atol=1e-5)
angle2=result2['x']
switch2,y2=trajectory2(angle2,equ2,equ1,Hfun,n,T,Nt,1e-7,40,40,complete=False)


fig, ax = plt.subplots()

plt.plot(switch1[:,0], switch1[:,1])
plt.plot(switch2[:,0], switch2[:,1])
plt.plot(equ1[0],equ1[1],'o', color='black', mfc='white', markersize=8)
plt.plot(equ2[0],equ2[1],'o', color='black', markersize=8)
plt.plot(equ3[0],equ3[1],'o', color='black', markersize=8)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

ax.text(
    0.13, 0.98, "$x_0$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
ax.text(
    0.55, 0.6, "$x_s$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

ax.text(
    0.98, 0.18, "$x_0\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


a1=1
a2=1
b1=1
b2=1
omega=0.5
k1=1
k2=1
m=4
xvar='x1 x2'
pvar='p1 p2'
xs=symbols(xvar)
ps=symbols(pvar)
rate=[a1*xs[0]**m/(omega**m+xs[0]**m)+ b1*omega**m/(omega**m+(xs[1])**m),k1*xs[0],a2*xs[1]**m/(omega**m+xs[1]**m)+ b2*omega**m/(omega**m+(xs[0])**m),k2*xs[1]]
react=[[1,0],[-1,0],[0,1],[0,-1]]
Hfun = 0
N=len(rate)
n=len(ps)
# define the Hamiltonian
for i in range(N):
    Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun

equ1=np.array(nsolve((Matrix(rate).T*Matrix(react)),(xs[0],xs[1]),(1,1))).astype(np.complex).real
equ2=np.array(nsolve((Matrix(rate).T*Matrix(react)),(xs[0],xs[1]),(3,0))).astype(np.complex).real
equ3=np.array(nsolve((Matrix(rate).T*Matrix(react)),(xs[0],xs[1]),(0,3))).astype(np.complex).real
equ4=np.array(nsolve((Matrix(rate).T*Matrix(react)),(xs[0],xs[1]),(1.5,0.5))).astype(np.complex).real
equ5=np.array(nsolve((Matrix(rate).T*Matrix(react)),(xs[0],xs[1]),(0.5,1.5))).astype(np.complex).real


equ1=np.array([equ1[0][0],equ1[1][0]])
equ2=np.array([equ2[0][0],equ2[1][0]])
equ3=np.array([equ3[0][0],equ3[1][0]])
equ4=np.array([equ4[0][0],equ4[1][0]])
equ5=np.array([equ5[0][0],equ5[1][0]])

# Plot vector field
x,y = np.meshgrid(np.linspace(0,2.3,15),np.linspace(0,2.3,10))
u = a1*x**m/(omega**m+x**m)+b1*omega**m/(omega**m+y**m)-k1*x
v = a2*y**m/(omega**m+y**m)+b2*omega**m/(omega**m+x**m)-k2*y

fig, ax = plt.subplots()

ax.streamplot(x,y,u,v, density = 1.9, color='grey')
plt.plot(equ1[0],equ1[1],'o', color='black', markersize=8)
plt.plot(equ2[0],equ2[1],'o', color='black', markersize=8)
plt.plot(equ3[0],equ3[1],'o', color='black', markersize=8)
plt.plot(equ4[0],equ4[1],'o', color='black', mfc='white', markersize=8)
plt.plot(equ5[0],equ5[1],'o', color='black', mfc='white', markersize=8)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.ylim((-0.05,2.3))
plt.xlim((-0.05,2.3))
ax.text(
    equ1[0]+0.15,equ1[1]+0.2, "$x_0\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
ax.text(
    equ2[0],equ2[1]+0.2, "$x_0\'\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

ax.text(
    equ3[0]+0.15,equ3[1], "$x_0$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
ax.text(
    equ4[0],equ4[1]+0.2, "$x_s\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

ax.text(
    equ5[0]+0.15,equ5[1], "$x_s$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))


# Next we would like to look at the switching trajectories 

result1b=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ5,Hfun,n,T,Nt,1e-7,40,40,True),tol=1e-3,atol=1e-5)
angle1b=result1b['x']
switch1b,y2=trajectory2(angle1b,equ3,equ5,Hfun,n,T,Nt,1e-7,40,40,complete=False)

result2b=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ1,equ4,Hfun,n,T,Nt,1e-5,40,40,True),tol=1e-3,atol=1e-5)
angle2b=result2b['x']
switch2b,y2=trajectory2(angle2b,equ1,equ4,Hfun,n,T,Nt,1e-5,40,40,complete=False)

result3=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ1,equ5,Hfun,n,T,Nt,1e-4,40,40,True),tol=1e-3,atol=1e-5)
angle3=result3['x']
switch3,y2=trajectory2(angle3,equ1,equ5,Hfun,n,T,Nt,1e-4,40,40,complete=False)

result4=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ2,equ4,Hfun,n,T,Nt,1e-5,40,40,True),tol=1e-3,atol=1e-5)
angle4=result4['x']
switch4,y2=trajectory2(angle4,equ2,equ4,Hfun,n,T,Nt,1e-5,40,40,complete=False)


fig, ax = plt.subplots()

plt.plot(switch1b[:,0], switch1b[:,1])
plt.plot(switch2b[:,0], switch2b[:,1])
plt.plot(switch3[:,0], switch3[:,1])
plt.plot(switch4[:,0], switch4[:,1])
plt.plot(equ1[0],equ1[1],'o', color='black', markersize=8)
plt.plot(equ2[0],equ2[1],'o', color='black', markersize=8)
plt.plot(equ3[0],equ3[1],'o', color='black', markersize=8)
plt.plot(equ4[0],equ4[1],'o', color='black', mfc='white', markersize=8)
plt.plot(equ5[0],equ5[1],'o', color='black', mfc='white', markersize=8)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')


ax.text(
    equ1[0]+0.15,equ1[1]+0.2, "$x_0\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
ax.text(
    equ2[0]-0.03,equ2[1]+0.22, "$x_0\'\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

ax.text(
    equ3[0]+0.15,equ3[1]-0.05, "$x_0$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
ax.text(
    equ4[0],equ4[1]+0.22, "$x_s\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

ax.text(
    equ5[0]+0.15,equ5[1], "$x_s$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


# Next we would like to compute the escape time for a set of parameters and do some simulations to check this

j=0
dim=20
taulim=1e10
b1_range=np.linspace(0.8,1.2,dim)
tauvec= np.zeros(dim)
tauvec2= np.zeros(dim)
for j in range(dim):
    a1=0
    a2=0
    b1=b1_range[j]
    b2=1
    omega=0.5
    k1=1
    k2=1
    m=4
    xvar='x1 x2'
    pvar='p1 p2'
    xs=symbols(xvar)
    ps=symbols(pvar)
    rate=[a1*xs[0]**m/(omega**m+xs[0]**m)+ b1*omega**m/(omega**m+(xs[1])**m),k1*xs[0],a2*xs[1]**m/(omega**m+xs[1]**m)+ b2*omega**m/(omega**m+(xs[0])**m),k2*xs[1]]
    react=[[1,0],[-1,0],[0,1],[0,-1]]
    Hfun = 0
    S=len(rate)
    n=len(ps)
    # define the Hamiltonian
    for i in range(S):
        Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun
    Hfun

    equ1=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.5,0.5))).astype(np.complex).real
    equ2=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(b1,0.01))).astype(np.complex).real
    equ3=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.01,b2))).astype(np.complex).real

    equ1=np.array([equ1[0][0],equ1[1][0]])
    equ2=np.array([equ2[0][0],equ2[1][0]])
    equ3=np.array([equ3[0][0],equ3[1][0]])

    N=20
    T=70
    Nt=30000
    tau=1e15
    tol=3e-3
    tol1=5e-4
    tol2=2e-4
    n=2

    while tau>taulim:
        nrep=0
        result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,2,T,Nt,1e-3,40,40, True),tol=1e-3,atol=1e-5)
        angle2=result2['x']
        y1,y2=trajectory2(angle2,equ3,equ1,Hfun,2,T,Nt,1e-3,40,40,complete=False)
        while result2['fun']>tol:
            print(nrep)
            result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,n,T,Nt,1e-4,40,40, True),tol=1e-3,atol=1e-5)
            angle2=result2['x']
            y1,y2=trajectory2(angle2,equ3,equ1,Hfun,n,T,Nt,1e-4,40,40,complete=False)
            if result2['fun']>tol1:
                result2a=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,n,T,Nt,5e-3,40,40, True),tol=1e-3,atol=1e-5)
                if result2['fun']>result2a['fun']:
                    result2=result2a
                angle2=result2['x']
                y1,y2=trajectory2(angle2,equ3,equ1,Hfun,n,T,Nt,5e-3,40,40,complete=False)
            nrep+=1
        k_s=y2[-1,0]
        k_0=y2[0,0]
        z0 = np.reshape(y2[0,1:],(n,n), order='F')
        detz_0=np.linalg.det(z0)
        zs = np.reshape(y2[-1,1:],(n,n), order='F')
        detz_s=np.linalg.det(zs)
        subst1 = [equ1[i] for i in range(n)]#xs
        subst2 = [0 for i in range(n)]#ps
        Hfundxdp = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),ps))
        eig_vals = np.linalg.eig(Hfundxdp(subst1, subst2))[0]
        # Take positive eigval:
        lambda_u=np.max(eig_vals)
        tau=k_0/k_s*np.sqrt(abs(detz_s)/detz_0)/lambda_u*np.pi*np.exp(N*y1[-1,4])
    tauvec[j]=tau
    # Other direction

    tau=1e15
    while tau>taulim:
        nrep=0
        result2r=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ2,equ1,Hfun,n,T,Nt,1e-4,40,40, True),tol=1e-3,atol=1e-5)
        angle2r=result2r['x']
        y1r,y2r=trajectory2(angle2r,equ2,equ1,Hfun,n,T,Nt,1e-4,40,40,complete=False)
        while result2r['fun']>tol:
            print(nrep)
            result2r=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ2,equ1,Hfun,n,T,Nt,1e-3,40,40, True),tol=1e-3,atol=1e-5)
            angle2r=result2r['x']
            y1r,y2r=trajectory2(angle2r,equ2,equ1,Hfun,n,T,Nt,1e-3,40,40,complete=False)
            if result2r['fun']>tol1:
                result2ra=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ2,equ1,Hfun,n,T,Nt,7e-4,40,40, True),tol=1e-3,atol=1e-5)
                if result2r['fun']>result2ra['fun']:
                    result2r=result2ra
                angle2r=result2r['x']
                y1r,y2r=trajectory2(angle2r,equ2,equ1,Hfun,n,T,Nt,7e-4,40,40,complete=False)
                if result2r['fun']>tol2:
                    result2rb=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ2,equ1,Hfun,n,T,Nt,2e-3,40,40, True),tol=1e-3,atol=1e-5)
                    if result2r['fun']>result2rb['fun']:
                        result2r=result2rb
                    angle2r=result2r['x']
                    y1r,y2r=trajectory2(angle2r,equ2,equ1,Hfun,n,T,Nt,2e-3,40,40,complete=False)
            nrep+=1
        k_s=y2r[-1,0]
        k_0=y2r[0,0]
        z0 = np.reshape(y2r[0,1:],(n,n), order='F')
        detz_0=np.linalg.det(z0)
        zs = np.reshape(y2r[-1,1:],(n,n), order='F')
        detz_s=np.linalg.det(zs)
        subst1 = [equ1[i] for i in range(n)]#xs
        subst2 = [0 for i in range(n)]#ps
        Hfundxdp = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),ps))
        eig_vals = np.linalg.eig(Hfundxdp(subst1, subst2))[0]
        # Take positive eigval:
        lambda_u=np.max(eig_vals)
        tau=k_0/k_s*np.sqrt(abs(detz_s)/detz_0)/lambda_u*np.pi*np.exp(N*y1r[-1,4])
    tauvec2[j]=tau
    print(j)
plt.plot(y1[:,0],y1[:,1])
plt.plot(b1_range,tauvec)
plt.plot(b1_range,tauvec2)
plt.yscale('log')
# And now we compare these results to our simulations


j=0
dim=20
b1_range=np.linspace(0.8,1.2,dim)
meanvec1 = np.zeros(dim)
varvec1 = np.zeros(dim)
for j in range(dim):
    a1=0
    a2=0
    b1=b1_range[j]
    b2=1
    omega=0.5
    k1=1
    k2=1
    m=4
    xvar='x1 x2'
    pvar='p1 p2'
    xs=symbols(xvar)
    ps=symbols(pvar)
    rate=[a1*xs[0]**m/(omega**m+xs[0]**m)+ b1*omega**m/(omega**m+(xs[1])**m),k1*xs[0],a2*xs[1]**m/(omega**m+xs[1]**m)+ b2*omega**m/(omega**m+(xs[0])**m),k2*xs[1]]
    react=[[1,0],[-1,0],[0,1],[0,-1]]
    Hfun = 0
    S=len(rate)
    n=len(ps)
    # define the Hamiltonian
    for i in range(S):
        Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun
    Hfun

    equ1=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.5,0.5))).astype(np.complex).real
    equ2=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(b1,0.01))).astype(np.complex).real
    equ3=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.01,b2))).astype(np.complex).real

    equ1=np.array([equ1[0][0],equ1[1][0]])
    equ2=np.array([equ2[0][0],equ2[1][0]])
    equ3=np.array([equ3[0][0],equ3[1][0]])
    N=20
    timeamount=100
    nx=equ3
    slopevec = equ3-equ2
    slope=abs(slopevec[0]/(slopevec[1]))
    intercept = -slope*equ1[0]+equ1[1]
    escape_t=[]
    for n in range(timeamount):
        i=0
        t=0
        nx=equ3
        startq=np.random.uniform(0,1,2000)
        uniforms=np.random.uniform(0,1,2000)
        uniformst=np.random.uniform(0,1,2000)
        loguniformst=np.log(uniformst)
        ind=0
        while t > -1:
            if i==2000:
                i=0
                startq=np.random.uniform(0,1,2000)
                uniforms=np.random.uniform(0,1,2000)
                uniformst=np.random.uniform(0,1,2000)
                loguniformst=np.log(uniformst)
            qx=nx
            if ind==0:
                rateplus1=b1*omega**m/(omega**m+(qx[1])**m)
                rateminus1=k1*qx[0]
                rateplus2= b2*omega**m/(omega**m+(qx[0])**m)
                rateminus2=k2*qx[1]
            if ind==1:
                rateplus2= b2*omega**m/(omega**m+(qx[0])**m)
                rateminus1=k1*qx[0]
            if ind==2:
                rateplus1=b1*omega**m/(omega**m+(qx[1])**m)
                rateminus2=k2*qx[1]
            w=rateplus1+rateminus1+rateplus2+rateminus2
            dt=-loguniformst[i]/w
            t+=dt
            if rateplus1/w>uniforms[i]:
                nx=np.array([nx[0]+1/N,nx[1]])
                ind=1
            elif (rateminus1+rateplus1)/w>uniforms[i]:
                if 1/N<qx[0]:
                    nx=np.array([nx[0]-1/N,nx[1]])
                    ind=1
                else:
                    ind=0
            elif (rateminus1+rateplus1+rateplus2)/w>uniforms[i]:
                nx=np.array([nx[0],nx[1]+1/N])
                ind=2
            else:
                if 1/N<qx[1]:
                    nx=np.array([nx[0],nx[1]-1/N])
                    ind=2
                else:
                    ind=0
            #condition for stopping:
            if qx[1]<slope*qx[0]+intercept:
                escape_t.append(t)
                break
            i+=1
    meanvec1[j] = np.mean(escape_t)
    varvec1[j] = np.var(escape_t)
    j+=1

# And the other escape time

j=0
dim=20
b1_range=np.linspace(0.8,1.2,dim)
meanvec2 = np.zeros(dim)
varvec2 = np.zeros(dim)
for j in range(dim):
    a1=0
    a2=0
    b1=b1_range[j]
    b2=1
    omega=0.5
    k1=1
    k2=1
    m=4
    xvar='x1 x2'
    pvar='p1 p2'
    xs=symbols(xvar)
    ps=symbols(pvar)
    rate=[a1*xs[0]**m/(omega**m+xs[0]**m)+ b1*omega**m/(omega**m+(xs[1])**m),k1*xs[0],a2*xs[1]**m/(omega**m+xs[1]**m)+ b2*omega**m/(omega**m+(xs[0])**m),k2*xs[1]]
    react=[[1,0],[-1,0],[0,1],[0,-1]]
    Hfun = 0
    S=len(rate)
    n=len(ps)
    # define the Hamiltonian
    for i in range(S):
        Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun
    Hfun

    equ1=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.5,0.5))).astype(np.complex).real
    equ2=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(b1,0.01))).astype(np.complex).real
    equ3=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.01,b2))).astype(np.complex).real

    equ1=np.array([equ1[0][0],equ1[1][0]])
    equ2=np.array([equ2[0][0],equ2[1][0]])
    equ3=np.array([equ3[0][0],equ3[1][0]])
    N=20
    timeamount=100
    nx=equ2
    slopevec = equ3-equ2
    slope=abs(slopevec[0]/(slopevec[1]))
    intercept = -slope*equ1[0]+equ1[1]
    escape_t=[]
    for n in range(timeamount):
        i=0
        t=0
        nx=equ2
        startq=np.random.uniform(0,1,2000)
        uniforms=np.random.uniform(0,1,2000)
        uniformst=np.random.uniform(0,1,2000)
        loguniformst=np.log(uniformst)
        ind=0
        while t > -1:
            if i==2000:
                i=0
                startq=np.random.uniform(0,1,2000)
                uniforms=np.random.uniform(0,1,2000)
                uniformst=np.random.uniform(0,1,2000)
                loguniformst=np.log(uniformst)
            qx=nx
            if ind==0:
                rateplus1=b1*omega**m/(omega**m+(qx[1])**m)
                rateminus1=k1*qx[0]
                rateplus2= b2*omega**m/(omega**m+(qx[0])**m)
                rateminus2=k2*qx[1]
            if ind==1:
                rateplus2= b2*omega**m/(omega**m+(qx[0])**m)
                rateminus1=k1*qx[0]
            if ind==2:
                rateplus1=b1*omega**m/(omega**m+(qx[1])**m)
                rateminus2=k2*qx[1]
            w=rateplus1+rateminus1+rateplus2+rateminus2
            dt=-loguniformst[i]/w
            t+=dt
            if rateplus1/w>uniforms[i]:
                nx=np.array([nx[0]+1/N,nx[1]])
                ind=1
            elif (rateminus1+rateplus1)/w>uniforms[i]:
                if 1/N<qx[0]:
                    nx=np.array([nx[0]-1/N,nx[1]])
                    ind=1
                else:
                    ind=0
            elif (rateminus1+rateplus1+rateplus2)/w>uniforms[i]:
                nx=np.array([nx[0],nx[1]+1/N])
                ind=2
            else:
                if 1/N<qx[1]:
                    nx=np.array([nx[0],nx[1]-1/N])
                    ind=2
                else:
                    ind=0
            #condition for stopping:
            if qx[1]>slope*qx[0]+intercept:
                escape_t.append(t)
                break
            i+=1
    meanvec2[j] = np.mean(escape_t)
    varvec2[j] = np.var(escape_t)
    j+=1


plt.plot(b1_range,meanvec1)
plt.plot(b1_range,meanvec2)
plt.plot(b1_range,tauvec)
plt.plot(b1_range,tauvec2)
plt.yscale('log')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylabel(r'$\tau$ (Escape time)')
plt.xlabel(r'$b_1$')


# Next we will make simulations to obtain the escape time
a1=0
a2=0
b1=1
b2=1
omega=0.5
k1=1
k2=1
m=4
xvar='x1 x2'
pvar='p1 p2'
xs=symbols(xvar)
ps=symbols(pvar)
rate=[a1*xs[0]**m/(omega**m+xs[0]**m)+ b1*omega**m/(omega**m+(xs[1])**m),k1*xs[0],a2*xs[1]**m/(omega**m+xs[1]**m)+ b2*omega**m/(omega**m+(xs[0])**m),k2*xs[1]]
react=[[1,0],[-1,0],[0,1],[0,-1]]
Hfun = 0
S=len(rate)
n=len(ps)
# define the Hamiltonian
for i in range(S):
    Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun
Hfun

equ1=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.5,0.5))).astype(np.complex).real
equ2=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(b1,0.01))).astype(np.complex).real
equ3=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.01,b2))).astype(np.complex).real

equ1=np.array([equ1[0][0],equ1[1][0]])
equ2=np.array([equ2[0][0],equ2[1][0]])
equ3=np.array([equ3[0][0],equ3[1][0]])

# We will now plot the distribution of the escape times and compare it to an exponential
simulations=1000
maxt=10000
nx=equ3
N=25
slopevec = equ3-equ2
slope=abs(slopevec[0]/(slopevec[1]))
intercept = -slope*equ1[0]+equ1[1]
escape_t=[]
#First a single simulation:
# for n in range(simulations):

for n in range(simulations):
    i=0
    t=0
    nx=equ3
    startq=np.random.uniform(0,1,2000*3)
    uniforms=np.random.uniform(0,1,2000*3)
    uniformst=np.random.uniform(0,1,2000*3)
    loguniformst=np.log(uniformst)
    while t > -1:
        if i==2000:
            i=0
            startq=np.random.uniform(0,1,2000*3)
            uniforms=np.random.uniform(0,1,2000*3)
            uniformst=np.random.uniform(0,1,2000*3)
            loguniformst=np.log(uniformst)
        qx=nx
        # print(nx)
        rateplus1=a1*qx[0]**m/(omega**m+qx[0]**m)+ b1*omega**m/(omega**m+(qx[1])**m)
        rateminus1=k1*qx[0]
        rateplus2=a2*qx[1]**m/(omega**m+qx[1]**m)+ b2*omega**m/(omega**m+(qx[0])**m)
        rateminus2=k2*qx[1]
        w1=rateplus1+rateminus1
        w2=rateplus2+rateminus2
        w=w1+w2
        dt=-loguniformst[i]/w
        t+=dt
        if w1/w>uniforms[i] and qx[0]<150:
            if rateplus1/w1>uniforms[2000+i] and qx[0]<150:
                nx=np.array([nx[0]+1/N,nx[1]])
            else:
                if 1/N<qx[0]:
                    nx=np.array([nx[0]-1/N,nx[1]])
        else:
            if rateplus2/w2>uniforms[2000*2+i]:
                nx=np.array([nx[0],nx[1]+1/N])
            else:
                if 1/N<qx[1]:
                    nx=np.array([nx[0],nx[1]-1/N])
        #condition for stopping:
        if qx[1]<slope*qx[0]+intercept:
            escape_t.append(t)
            break
        i+=1
lambdaval=1/np.mean(escape_t)
range_x=np.linspace(0,np.max(escape_t),1000)
exponentialdist = lambdaval*np.exp(-range_x*lambdaval)
plt.hist(escape_t,bins=130, density=True)
plt.plot(range_x,exponentialdist)
plt.xlabel(r'$\tau$')
plt.ylabel('Probability')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

# Now we compute a range of system sizes

n_N=20
systemsize = np.linspace(5,40,n_N)
meanvec = np.zeros(n_N)
varvec = np.zeros(n_N)
j=0
for N in systemsize:
    print(N)
    timeamount=200
    nx=equ3
    slopevec = equ3-equ2
    slope=abs(slopevec[0]/(slopevec[1]))
    intercept = -slope*equ1[0]+equ1[1]
    escape_t=[]
    for n in range(timeamount):
        i=0
        t=0
        nx=equ3
        startq=np.random.uniform(0,1,2000)
        uniforms=np.random.uniform(0,1,2000)
        uniformst=np.random.uniform(0,1,2000)
        loguniformst=np.log(uniformst)
        while t > -1:
            if i==2000:
                i=0
                startq=np.random.uniform(0,1,2000)
                uniforms=np.random.uniform(0,1,2000)
                uniformst=np.random.uniform(0,1,2000)
                loguniformst=np.log(uniformst)
            qx=nx
            # print(nx)
            rateplus1=a1*qx[0]**m/(omega**m+qx[0]**m)+ b1*omega**m/(omega**m+(qx[1])**m)
            rateminus1=k1*qx[0]
            rateplus2=a2*qx[1]**m/(omega**m+qx[1]**m)+ b2*omega**m/(omega**m+(qx[0])**m)
            rateminus2=k2*qx[1]
            w=rateplus1+rateminus1+rateplus2+rateminus2
            dt=-loguniformst[i]/w
            t+=dt
            if rateplus1/w>uniforms[i]:
                nx=np.array([nx[0]+1/N,nx[1]])
            elif (rateminus1+rateplus1)/w>uniforms[i]:
                if 1/N<qx[0]:
                    nx=np.array([nx[0]-1/N,nx[1]])
            elif (rateminus1+rateplus1+rateplus2)/w>uniforms[i]:
                nx=np.array([nx[0],nx[1]+1/N])
            else:
                if 1/N<qx[1]:
                    nx=np.array([nx[0],nx[1]-1/N])
            #condition for stopping:
            if qx[1]<slope*qx[0]+intercept:
                escape_t.append(t)
                break
            i+=1
    meanvec[j] = np.mean(escape_t)
    varvec[j] = np.var(escape_t)
    j+=1


np.std(escape_t)
plt.hist(escape_t)
np.mean(escape_t)
plt.errorbar(systemsize,meanvec, np.sqrt(varvec/200),marker='.', capsize=3)

N=20
T=70
Nt=30000
n=2
result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,2,T,Nt,1e-9,40,40, True),tol=1e-3,atol=1e-5)
angle2=result2['x']
y1,y2=trajectory2(angle2,equ3,equ1,Hfun,2,T,Nt,1e-4,40,40,complete=False)
sx1 = y1[-1,4]
result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,2,T,Nt,1e-4,40,40, True),tol=1e-3,atol=1e-5)
angle2=result2['x']
y1,y2=trajectory2(angle2,equ3,equ1,Hfun,2,T,Nt,1e-4,40,40,complete=False)
sx = y1[-1,4]
k_s=y2[-1,0]
k_0=y2[0,0]
z0 = np.reshape(y2[0,1:],(n,n), order='F')
detz_0=np.linalg.det(z0)
zs = np.reshape(y2[-1,1:],(n,n), order='F')
detz_s=np.linalg.det(zs)
subst1 = [equ1[i] for i in range(n)]#xs
subst2 = [0 for i in range(n)]#ps
Hfundxdp = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),ps))
eig_vals = np.linalg.eig(Hfundxdp(subst1, subst2))[0]
# Take positive eigval:
lambda_u=np.max(eig_vals)
tau=k_0/k_s*np.sqrt(abs(detz_s)/detz_0)/lambda_u*np.pi*np.exp(systemsize*(sx))#+0.09))
tau1=k_0/k_s*np.sqrt(abs(detz_s)/detz_0)/lambda_u*np.pi*np.exp(systemsize*(sx1))#+0.09))

plt.plot(systemsize, tau1)
plt.plot(systemsize, tau)
plt.plot(systemsize,meanvec)
plt.yscale('log')
plt.xlabel(r'$N$ (System size)')
plt.ylabel(r'$\tau$ (Escape time)')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)






# Make simulations with Gillespie
#Simulation with time
N=20
maxt=1000
a1=0
a2=0
b1=1
b2=1
omega=0.5
k1=1
k2=1
m=4
s=3
#exponential rate is 1 as maximum (assuming w0+w1>1), so can take 3/2*maxt reactions to generate the uniforms
simulations=100
startq=np.random.uniform(0,1,2*simulations)
uniforms=np.random.uniform(0,1,2*simulations*s*maxt)
uniformst=np.random.uniform(0,1,2*simulations*s*maxt)
loguniformst=np.log(uniformst)
sim = equ3
nx=equ3
#First a single simulation:
i=0
t=0
while t<maxt:
    qx=nx
    rateplus1=a1*qx[0]**m/(omega**m+qx[0]**m)+ b1*omega**m/(omega**m+(qx[1])**m)
    rateminus1=k1*qx[0]
    rateplus2=a2*qx[1]**m/(omega**m+qx[1]**m)+ b2*omega**m/(omega**m+(qx[0])**m)
    rateminus2=k2*qx[1]
    w1=rateplus1+rateminus1
    w2=rateplus2+rateminus2
    w=w1+w2
    dt=-loguniformst[i]/w
    t=t+dt
    if w1/w>uniforms[maxt+i] and qx[0]<150:
        if rateplus1/w1>uniforms[maxt*2+i] and qx[0]<150:
            nx=np.array([nx[0]+1/N,nx[1]])
        else:
            if 1/N<qx[0]:
                nx=np.array([nx[0]-1/N,nx[1]])
    else:
        if rateplus2/w2>uniforms[maxt*3+i]:
            nx=np.array([nx[0],nx[1]+1/N])
        else:
            if 1/N<qx[1]:
                nx=np.array([nx[0],nx[1]-1/N])
    sim= np.vstack((sim,nx))
    i+=1
# Plot of switching path and simulations
plt.plot(sim[:,0], sim[:,1],'.')
plt.plot(y1[:,0],y1[:,1]) # Need to compute these for the parameters considered
plt.plot(y1r[:,0],y1r[:,1])
plt.ylabel('Gene 2')
plt.xlabel('Gene 1')
plt.scatter(equ1[0],equ1[1], color='red', marker='s',zorder=3)
plt.scatter(equ2[0],equ2[1], color='red', marker='s',zorder=3)
plt.scatter(equ3[0],equ3[1], color='red', marker='s',zorder=3)
plt.annotate(r'$X_1$',(0.57,0.57),zorder=3)
plt.annotate(r'$X_2$',(equ2[0],0.175),zorder=3)
plt.annotate(r'$X_3$',(0,1.031),zorder=3)

# Now we want to do a histogram of the switching paths
N=42
interval=250
timeamount=70
nx=equ3
slopevec = equ3-equ2
slope=abs(slopevec[0]/(slopevec[1]))
intercept = -slope*equ1[0]+equ1[1]
vectpaths = np.empty(2)
escape_t=[]
for n in range(timeamount):
    print(n)
    i=0
    t=0
    nx=equ3
    sim=equ3
    uniforms=np.random.uniform(0,1,2000)
    uniformst=np.random.uniform(0,1,2000)
    loguniformst=np.log(uniformst)
    while t > -1:
        if i==2000:
            i=0
            startq=np.random.uniform(0,1,2000)
            uniforms=np.random.uniform(0,1,2000)
            uniformst=np.random.uniform(0,1,2000)
            loguniformst=np.log(uniformst)
        qx=nx
        rateplus1= b1*omega**m/(omega**m+(qx[1])**m)
        rateminus1=k1*qx[0]
        rateplus2= b2*omega**m/(omega**m+(qx[0])**m)
        rateminus2=k2*qx[1]
        w=rateplus1+rateminus1+rateplus2+rateminus2
        dt=-loguniformst[i]/w
        t+=dt
        if rateplus1/w>uniforms[i]:
            nx=np.array([nx[0]+1/N,nx[1]])
        elif (rateminus1+rateplus1)/w>uniforms[i]:
            if 1/N<qx[0]:
                nx=np.array([nx[0]-1/N,nx[1]])
        elif (rateminus1+rateplus1+rateplus2)/w>uniforms[i]:
            nx=np.array([nx[0],nx[1]+1/N])
        else:
            if 1/N<qx[1]:
                nx=np.array([nx[0],nx[1]-1/N])
        #condition for stopping:
        sim= np.vstack((sim,nx))
        if qx[1]<slope*qx[0]+intercept:
            escape_t.append(t)
            if len(sim)<interval:
                vectpaths= np.vstack((vectpaths,sim))
            else:
                vectpaths= np.vstack((vectpaths,sim[-interval:,:]))
            break
        i+=1
len(vectpaths)
plt.hist2d(vectpaths[1:,0], vectpaths[1:,1], bins=(17,23))


# the other side
N=42
interval=250
timeamount=70
nx=equ2
slopevec = equ3-equ2
slope=abs(slopevec[0]/(slopevec[1]))
intercept = -slope*equ1[0]+equ1[1]
vectpaths2 = np.empty(2)
escape_t=[]
for n in range(timeamount):
    print(n)
    i=0
    t=0
    nx=equ2
    sim=equ2
    uniforms=np.random.uniform(0,1,2000)
    uniformst=np.random.uniform(0,1,2000)
    loguniformst=np.log(uniformst)
    while t > -1:
        if i==2000:
            i=0
            startq=np.random.uniform(0,1,2000)
            uniforms=np.random.uniform(0,1,2000)
            uniformst=np.random.uniform(0,1,2000)
            loguniformst=np.log(uniformst)
        qx=nx
        rateplus1= b1*omega**m/(omega**m+(qx[1])**m)
        rateminus1=k1*qx[0]
        rateplus2= b2*omega**m/(omega**m+(qx[0])**m)
        rateminus2=k2*qx[1]
        w=rateplus1+rateminus1+rateplus2+rateminus2
        dt=-loguniformst[i]/w
        t+=dt
        if rateplus1/w>uniforms[i]:
            nx=np.array([nx[0]+1/N,nx[1]])
        elif (rateminus1+rateplus1)/w>uniforms[i]:
            if 1/N<qx[0]:
                nx=np.array([nx[0]-1/N,nx[1]])
        elif (rateminus1+rateplus1+rateplus2)/w>uniforms[i]:
            nx=np.array([nx[0],nx[1]+1/N])
        else:
            if 1/N<qx[1]:
                nx=np.array([nx[0],nx[1]-1/N])
        #condition for stopping:
        sim= np.vstack((sim,nx))
        if qx[1]>slope*qx[0]+intercept:
            escape_t.append(t)
            if len(sim)<interval:
                vectpaths2= np.vstack((vectpaths2,sim))
            else:
                vectpaths2= np.vstack((vectpaths2,sim[-interval:,:]))
            break
        i+=1
len(vectpaths2)
plt.hist2d(vectpaths2[1:,0], vectpaths2[1:,1], bins=(17,23))


vectpaths3=vectpaths
vectpaths4=vectpaths2

fig, ax = plt.subplots()
plt.hist2d(np.hstack((vectpaths[1:,0],vectpaths2[1:,0])), np.hstack((vectpaths[1:,1],vectpaths2[1:,1])), bins=(30,30))
plt.xlim((0.01,1.2))
plt.ylim((0.01,1.2))
plt.plot(switch1[:,0], switch1[:,1], color='orangered')
plt.plot(switch2[:,0], switch2[:,1], color='orangered')
plt.plot(equ1[0],equ1[1],'o', color='black', mfc='white', markersize=8)
plt.plot(equ2[0],equ2[1],'o', color='black', markersize=8)
plt.plot(equ3[0],equ3[1],'o', color='black', markersize=8)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

ax.text(
    0.17, 1.1, "$x_0$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))
ax.text(
    0.6, 0.65, "$x_s$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))

ax.text(
    1, 0.23, "$x_0\'$", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="square", fc='white'))




# We will now compare the formula to simulations up to a system size of 90. These were done independently so 
# you will need to load the datafiles from the repository
A=np.load('datafile34.npz')['escapetimes']
A=np.append(A,np.load('datafile261.npz')['escapetimes'], axis=1)
A=np.append(A,np.load('datafile286.npz')['escapetimes'], axis=1)
A=np.append(A,np.load('datafile366.npz')['escapetimes'], axis=1)
A=np.append(A,np.load('datafile404.npz')['escapetimes'], axis=1)
A=np.append(A,np.load('datafile800.npz')['escapetimes'], axis=1)
A=np.append(A,np.load('datafile888.npz')['escapetimes'], axis=1)
A=np.append(A,np.load('datafile931.npz')['escapetimes'], axis=1)


a1=0
a2=0
b1=1
b2=1
omega=0.5
k1=1
k2=1
m=4
#10 2 1
#10 1 0.5
xvar='x1 x2'
pvar='p1 p2'
xs=symbols(xvar)
ps=symbols(pvar)
rate=[a1*xs[0]**m/(omega**m+xs[0]**m)+ b1*omega**m/(omega**m+(xs[1])**m),k1*xs[0],a2*xs[1]**m/(omega**m+xs[1]**m)+ b2*omega**m/(omega**m+(xs[0])**m),k2*xs[1]]
react=[[1,0],[-1,0],[0,1],[0,-1]]
Hfun = 0
S=len(rate)
n=len(ps)
# define the Hamiltonian
for i in range(S):
    Hfun = rate[i]*(exp(Matrix(ps).dot(Matrix(react[i])))-1) + Hfun
Hfun

equ1=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.5,0.5))).astype(np.complex).real
equ2=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(b1,0.01))).astype(np.complex).real
equ3=np.array(nsolve((rate[0]-rate[1],rate[2]-rate[3]),xs,(0.01,b2))).astype(np.complex).real

equ1=np.array([equ1[0][0],equ1[1][0]])
equ2=np.array([equ2[0][0],equ2[1][0]])
equ3=np.array([equ3[0][0],equ3[1][0]])
N=20
T=70
Nt=30000
tau=1e15
n=2
result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,2,T,Nt,1e-9,40,40, True),tol=1e-3,atol=1e-5)
angle2=result2['x']
y1,y2=trajectory2(angle2,equ3,equ1,Hfun,2,T,Nt,1e-4,40,40,complete=False)
sx = y1[-1,4]
result2=scp.optimize.differential_evolution(trajectory2,bounds=[(0,2*np.pi),],args=(equ3,equ1,Hfun,2,T,Nt,1e-4,40,40, True),tol=1e-3,atol=1e-5)
angle2=result2['x']
y1,y2=trajectory2(angle2,equ3,equ1,Hfun,2,T,Nt,1e-4,40,40,complete=False)
sx = y1[-1,4]
k_s=y2[-1,0]
k_0=y2[0,0]
z0 = np.reshape(y2[0,1:],(n,n), order='F')
detz_0=np.linalg.det(z0)
zs = np.reshape(y2[-1,1:],(n,n), order='F')
detz_s=np.linalg.det(zs)
subst1 = [equ1[i] for i in range(n)]#xs
subst2 = [0 for i in range(n)]#ps
Hfundxdp = lambdify([xs,ps],derive_by_array(derive_by_array(Hfun,xs),ps))
eig_vals = np.linalg.eig(Hfundxdp(subst1, subst2))[0]
# Take positive eigval:
lambda_u=np.max(eig_vals)
tau=k_0/k_s*np.sqrt(abs(detz_s)/detz_0)/lambda_u*np.pi*np.exp(systemsize*(sx))


meantimes=np.mean(A,axis=1)
systemsize = np.linspace(10,120,12)
plt.plot(systemsize[:9],meantimes[:9])
plt.plot(systemsize[:9],tau[:9])
plt.yscale('log')
plt.xlabel(r'$N$ (System size)')
plt.ylabel(r'$\tau$ (Escape time)')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


