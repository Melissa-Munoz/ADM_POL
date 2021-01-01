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
QIS = 0.0
UIS = 0.0
thetaIS = 0.0


#Calling ADM
#-------------------------------------------------------------------------------
phi = np.linspace(0.,1.,50) #rotational phase
Nx = Ny = Nz = 50 #grid size 
#Point light source (faster code, but can significantly overestimate the linear polarisation)
outp = ADM.POLp(phi, inc, beta, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bstar, delta, QIS, UIS, thetaIS)
#Finite star (slower, but perhaps more accurate)
out = ADM.POL(phi, inc, beta, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bstar, delta, QIS, UIS, thetaIS)


#Plotting phased Stokes Q and U curves
#-------------------------------------------------------------------------------
fig, ax = plt.subplots(2,figsize=(9,6),sharey=True)
ax[0].plot(phi, outp[0],'k')
ax[0].plot(phi+1, outp[0],'k')
ax[0].plot(phi-1, outp[0],'k')
ax[1].plot(phi, outp[1],'k')
ax[1].plot(phi+1, outp[1],'k')
ax[1].plot(phi-1, outp[1],'k')
ax[0].set_ylabel('Q [%]',fontsize=14)
ax[1].set_ylabel('U [%]',fontsize=14)
ax[1].set_xlabel('Rotational phase',fontsize=14)
ax[0].set_xlim([-0.5,1.5])
ax[1].set_xlim([-0.5,1.5])
plt.show()

plt.figure(figsize=(6,6))
plt.plot(outp[0],outp[1])
plt.plot(out[0],out[1])
plt.ylabel('Q [%]',fontsize=14)
plt.xlabel('U [%]',fontsize=14)
plt.show()

