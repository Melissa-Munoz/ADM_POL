import numpy as np
import matplotlib.pyplot as plt
import ADM


#Physical properties of the star
#-------------------------------------------------------------------------------
#Effective temperature, Teff, is in Kelvins
#Stellar mass, Mstar, is in solar mass
#Stellar radius, Rstar, is in solar radius
#Terminal velocity, Vinf, is in km/s
#Mass-loss rate, Mdot, is in solar mass per year
#Polar field strength, Bstar, is in Gauss
Teff = 35000.0 
Mstar = 30.0
Rstar = 10.
Vinf = 2500.0
Mdot = 10**(-6.0)
Bstar = 2500.0


#Geometric angles
#-------------------------------------------------------------------------------
#Inclination angle, inc, in degrees
#Magnetic obliquity, beta, in degrees
inc = 30.
beta = 60.
A = inc+beta
B = np.abs(inc-beta)


#Extra parameters
#-------------------------------------------------------------------------------
#Smoothing length, delta
#Interstellar polarisation, QIS and UIS
#Referance angle, thetaIS
delta = 0.1 
QIS = 0.
UIS = 0.
thetaIS = 0.


#Calling ADM
#-------------------------------------------------------------------------------
phi = np.linspace(0.,1.,25) #rotational phase
Nx = Ny = Nz = 25 #grid size 

#Point light source (faster code, but can significantly overestimate the linear polarisation)
outp = ADM.POLp(phi, inc, beta, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bstar, delta, QIS, UIS, thetaIS)

#Finite star (slower, but perhaps more accurate)
out = ADM.POL(phi, inc, beta, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bstar, delta, QIS, UIS, thetaIS)


#Plotting phased Stokes Q and U curves
#-------------------------------------------------------------------------------

fig, ax = plt.subplots(2, figsize=(9,6),sharex=True)
ax[0].plot(phi, outp[0],'k',label='Point light source')
ax[0].plot(phi+1, outp[0],'k')
ax[0].plot(phi-1, outp[0],'k')
ax[1].plot(phi, outp[1],'k')
ax[1].plot(phi+1, outp[1],'k')
ax[1].plot(phi-1, outp[1],'k')
ax[0].plot(phi, out[0],'--k',label='Finite star')
ax[0].plot(phi+1, out[0],'--k')
ax[0].plot(phi-1, out[0],'--k')
ax[1].plot(phi, out[1],'--k')
ax[1].plot(phi+1, out[1],'--k')
ax[1].plot(phi-1, out[1],'--k')
ax[0].legend()
ax[0].set_ylabel('Q [%]',fontsize=14)
ax[1].set_ylabel('U [%]',fontsize=14)
ax[1].set_xlabel('Rotational phase',fontsize=14)
ax[0].set_xlim([-0.5,1.5])
plt.show()


plt.figure(figsize=(6,6))
plt.plot(outp[0],outp[1],'k',label='Point light source')
plt.plot(out[0],out[1],'--k',label='Finite star')
plt.legend()
plt.ylabel('U [%]',fontsize=14)
plt.xlabel('Q [%]',fontsize=14)
plt.ylim([-0.8,0.8])
plt.xlim([-0.8,0.8])
plt.show()

Q=outp[0]
U=outp[1]
amp1 = (np.max(Q)-np.min(Q) + np.max(U)-np.min(U))/4.

Q=out[0]
U=out[1]
amp2 = (np.max(Q)-np.min(Q) + np.max(U)-np.min(U))/4.

print amp1/amp2, amp2/amp1

