#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ADM-based Stokes Q ans U curves synthesis 
# Melissa Munoz
# Updated Dec 2020
#
# See publication Munoz et al. 2020, in prep
# See also Owocki et al. 2006 for more details on the ADM formalism
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#-------------------------------------------------------------------------------
# Library import ---------------------------------------------------------------
#-------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import newton
from scipy import interpolate
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt
import time 
from scipy.integrate import simps


#-------------------------------------------------------------------------------
# Some constants ---------------------------------------------------------------
#-------------------------------------------------------------------------------

G = 6.67408*10**(-8)
eV = 1.602*10**(-12)
c = 2.99792458*10**(10)	 
h= 6.6260755*10**(-27)	 
kb = 1.380658*10**(-16)
eV = 1.6021772*10**(-12)	
me = 9.1093897*10**(-28)
mp = 1.6726*10**(-24)
mH = 1.6733*10**(-24)
e = 4.8032068*10**(-10)
alphae = (1.+0.7)/2.#0.5
sigmat = 6.6524*10**(-25)
sigma0=sigmat*3./(16.*np.pi)
Msol = 1.99*10**(33)
Rsol = 6.96*10**(10)
Lsol = 3.9*10**(33)



#-------------------------------------------------------------------------------
# Line-of-sight angle ----------------------------------------------------------
#-------------------------------------------------------------------------------

#With degenerancy between inclination and obliquity
def csalpha(phi, beta, inc):
	return np.sin(beta)*np.cos(phi)*np.sin(inc)+np.cos(beta)*np.cos(inc)

#To avoid the degenrancy, the Independant paramters A and B can be used, where
#A = inc + beta,
#B = |inc - beta|
def csalpha2(phi, A, B):
	return 0.5*(np.cos(B)*(1.+np.cos(phi)) + np.cos(A)*(1.-np.cos(phi)) )



#-------------------------------------------------------------------------------
# ADM auxiliary equations ------------------------------------------------------
#-------------------------------------------------------------------------------

# Wind upflow
#-------------------------------------------------------------------------------

def w(r):
	return 1.-1./r

def vw(r,vinf):
	return vinf*(1.-1./r)

def rhow(r,mu):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/((r- 1.)*(4.*r - 3. + 3.*mu**2))*(1./r)**(3./2.)


# Hot post-shock 
#-------------------------------------------------------------------------------

def wh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws/4.*(Th(rs,mu,mus,Tinf,Teff)/Ts)*(B(r,mu,mus))

def vh(r,rs,mu,mus,Tinf,Teff,vinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws*vinf/4.*Th(rs,mu,mus,Tinf,Teff)/Ts*np.sqrt((1.+3.*mu**2)/(1.+3.*mus**2))*(rs/r)**3

def g(mu):
	return np.abs(mu - mu**3 + 3.*mu**5/5. - mu**7/7.)

def TTh(rs,mu,mus,Tinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return Ts*(g(mu)/g(mus))**(1./3.)

def Th(rs,mu,mus,Tinf,Teff):
	return np.maximum(TTh(rs,mu,mus,Tinf),Teff)

def rhoh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return 4.*rhow(rs,mus)*Ts/Th(rs,mu,mus,Tinf,Teff)

# Cooled downflow 
#-------------------------------------------------------------------------------

def wc(r,mu):
	return np.abs(mu)*np.sqrt(1./r)

def vc(r,mu,ve):
	return np.abs(mu)*np.sqrt(1./r)*ve

def rhoc(r,mu,delta):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/(np.sqrt(mu**2+delta**2/r**2)*(4.*r - 3. + 3.*mu**2))*(1./r)**(2.)

def f(mus,mustar,chiinf):
	rm = 1./(1.-mustar**2)
	rs = rm*(1.-mus**2)
	ws = w(rs)
	ggmus = chiinf/(6.*mustar)*(1.+3.*mustar**2)/(1.+3.*mus**2)*(ws*rs/rm)**4*(rs)**2
	gmus = g(mus)
	return (gmus-ggmus) 



#-------------------------------------------------------------------------------
# ADM --------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def admCAL(Nx, Ny, Nz, RA, Rc, Teff, Tinf, chiinf, delta ):

	#Defining magnetosphere grid size	
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)	
	X=XX[Nx:NNx]
	Y=YY[Ny:NNy]
	Z=ZZ[Nz:NNz]
	
	#Defining density grids of each component
	Rhoh=np.zeros([Nz,Nx,Ny])
	Rhow=np.zeros([Nz,Nx,Ny])
	Rhoc=np.zeros([Nz,Nx,Ny])

	#Last closed loop
	mustar_RA = np.sqrt(1.-1./RA)
	mustars_RA = np.linspace(0.01,mustar_RA,Nz)
	mus_RA = np.zeros(Nz)
	rs_RA = np.zeros(Nz)
	r_RA = (1.-mustars_RA**2)/(1-mustar_RA**2)
	for i in range(Nz):
		try:
			tempmus = newton(f, 0.3, args=(mustars_RA[i],chiinf,))
		except RuntimeError:	
			tempmus=0.
			#print 'error LC'
		mus_RA[i]=np.abs(tempmus)
		rs_RA[i]=(1.-mus_RA[i]**2)/(1.-mustars_RA[i]**2)
	fs=interp1d( mustars_RA,mus_RA,bounds_error=False, fill_value=0. )

	#Compute ADM in first octant of the magnetosphere
	#Velocity and temperature calculations are commented out because they are not required for the light curve synthesis
	for i in range(0,Nx):
		for j in range(0,Ny):
			p=np.sqrt(X[i]**2+Y[j]**2)
			for k in range(0,Nz):
				r=np.sqrt(p**2+Z[k]**2)
				mu=Z[k]/r
				rRA=(1.-mu**2)/(1-mustar_RA**2)
				if r > 1.01:
					mustar=np.sqrt(1.-(1.-mu**2)/r)
					rm = 1./(1.-mustar**2)
					mus=fs(mustar)
					'''
					try:
						tempmus = newton(f, 0.3, args=(mustar,chiinf,))
					except RuntimeError:	
						tempmus=0.
						#print 'error LC'
					mus = np.abs(tempmus)
					'''
					rs = rm*(1.-mus**2)
					Rhow[k,i,j]=rhow(r,mu)
					#Vw[k,i,j]=w(r)
					#tw[k,i,j]=Teff		
					if r < rRA:
						Rhoc[k,i,j]=rhoc(r,mu,delta)
						#Vc[k,i,j]=wc(r,mu)
						#tc[k,i,j]=Teff
						if r > rs and rs > 1.01 :
							Rhoh[k,i,j]=rhoh(r,rs,mu,mus,Tinf,Teff)
							#Vh[k,i,j]=wh(r,rs,mu,mus,Tinf,Teff)
							#th[k,i,j]=Th(rs,mu,mus,Tinf,Teff)
	
	#Transpose density in remaining octants (axial symmetry)
	Rhoh=np.concatenate([Rhoh[::-1,:,:],Rhoh],axis=0)
	Rhoh=np.concatenate([Rhoh[:,::-1,:],Rhoh],axis=1)
	Rhoh=np.concatenate([Rhoh[:,:,::-1],Rhoh],axis=2)
	Rhoc=np.concatenate([Rhoc[::-1,:,:],Rhoc],axis=0)
	Rhoc=np.concatenate([Rhoc[:,::-1,:],Rhoc],axis=1)
	Rhoc=np.concatenate([Rhoc[:,:,::-1],Rhoc],axis=2)
	Rhow=np.concatenate([Rhow[::-1,:,:],Rhow],axis=0)
	Rhow=np.concatenate([Rhow[:,::-1,:],Rhow],axis=1)
	Rhow=np.concatenate([Rhow[:,:,::-1],Rhow],axis=2)	
	
	return [Rhow, Rhoh, Rhoc]		



#-------------------------------------------------------------------------------
# RT ---------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#Point light source approximation
#-------------------------------------------------------------------------------
def POLp(phi, inc, beta, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bd, delta, QIS, UIS, thetaIS):
	
	#Defining phase grid size
	PH=len(phi)

	#Defining magnetosphere grid size
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Conversion of stellar properties into cgs 
	mdot=Mdot*Msol/(365.*24*3600.)
	vinf=Vinf*100000.
	rstar=Rstar*Rsol
	mstar=Mstar*Msol
	ve = np.sqrt(2.*G*mstar/rstar)
	Ve = np.sqrt(2.*G*mstar/rstar)/100000.
	rhowstar = mdot/(4.*np.pi*rstar**2*vinf)
	rhocstar = rhowstar*vinf/ve

	#Some scalling relations. see Owocki 2016
	chiinf = 0.034*(vinf/10.**8)**4*(rstar/10**12)/(Mdot/10**(-6))
	Tinf = 14*10**6*(vinf/10.**8)**2

	#Computing the Alfven radius and closure radius
	Beq=Bd/2.
	eta=(Beq)**2*rstar**2/(mdot*vinf)
	RA=0.3+(eta+0.25)**(0.25)
	Rc = RA # This can be changed occording to the user

	#ADM output (the magnetosphere density components)
	admOUT=admCAL(Nx, Ny, Nz, RA, Rc, Teff, Tinf, chiinf, delta )
	Rhow=admOUT[0]
	Rhoh=admOUT[1]
	Rhoc=admOUT[2]
	rhoh=Rhoh*rhowstar
	rhoc=Rhoc*rhocstar
	rhow=Rhow*rhowstar

	#Electron density
	RHO=rhoh+rhoc+rhow	
	ne=RHO*alphae/mp

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)
	dX=np.abs(XX[0]-XX[1])
	dY=np.abs(YY[0]-YY[1])
	dZ=np.abs(ZZ[0]-ZZ[1])
	dx=dX*Rstar*Rsol
	dy=dY*Rstar*Rsol
	dz=dZ*Rstar*Rsol

	#Creating 3D meshgrids
	Z_grid, X_grid, Y_grid = np.meshgrid( ZZ, XX, YY, indexing='xy')
	R_grid = np.sqrt( Z_grid**2 + X_grid**2 + Y_grid**2 )
	MU_grid = X_grid/R_grid

	#Depolarisation factor (see Fox 1991)
	D=np.ones([NNz,NNx,NNy])	#For a point light source
	#D=np.sqrt(1.-1./R_grid**2) #For a finite star

	#Removing data 'inside' star and outside closure radius
	RHO[ R_grid < 1.0 ] = 0. 
	RHO[ R_grid > Rc ] = 0. 
	D[ R_grid < 1.0 ] = 0.
	D[ R_grid > Rc ] = 0.
	MU_grid[ R_grid < 1.0 ] = 0.
	MU_grid[ R_grid > Rc ] = 0.

	#Computation of integral moments
	tau0 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
	taugamma0 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*MU_grid**2,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)

	#Calculation of Stokes Q and U curves
	amp = (tau0-3.*taugamma0)*100.
	QU = QUp_(phi, amp, inc, beta, thetaIS, QIS, UIS)

	return QU


def QUp_(phi, amp, inc, beta, thetaIS, QIS, UIS):
	inc = np.radians(inc)
	beta = np.radians(beta)
	lda = phi*2.*np.pi

	q0= amp*np.sin(inc)**2*(3.*np.cos(beta)**2-1.)*np.cos(thetaIS) + QIS #QIS*np.cos(thetaIS) - UIS*np.sin(thetaIS) +
	q1= 2.*amp*np.sin(inc)*np.sin(2.*beta)*np.sin(thetaIS) 
	q2= amp*np.sin(2.*inc)*np.sin(2.*beta)*np.cos(thetaIS)
	q3=-amp*(1.+np.cos(inc)**2)*np.sin(beta)**2*np.cos(thetaIS)
	q4= 2.*amp*np.cos(inc)*np.sin(beta)**2*np.sin(thetaIS)
	Q = q0 + q1*np.cos(lda) + q2*np.sin(lda) + q3*np.cos(2.*lda) +  q4*np.sin(2.*lda)

	u0= amp*np.sin(inc)**2*(3.*np.cos(beta)**2-1)*np.sin(thetaIS) + UIS #QIS*np.sin(thetaIS) + UIS*np.cos(thetaIS) 
	u1=-2.*amp*np.sin(inc)*np.sin(2.*beta)*np.cos(thetaIS) 
	u2= amp*np.sin(2.*inc)*np.sin(2.*beta)*np.sin(thetaIS)
	u3=-amp*(1.+np.cos(inc)**2)*np.sin(beta)**2*np.sin(thetaIS)
	u4=-2.*amp*np.cos(inc)*np.sin(beta)**2*np.cos(thetaIS)
	U = u0 + u1*np.cos(lda) + u2*np.sin(lda) + u3*np.cos(2.*lda) +  u4*np.sin(2.*lda)
	return [Q,U]


#Finite light source correction
#-------------------------------------------------------------------------------

def POL(phi, inc, beta, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bd, delta, QIS, UIS, thetaIS):
	
	#Defining phase grid size
	PH=len(phi)

	#Defining magnetosphere grid size
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Conversion of stellar properties into cgs 
	mdot=Mdot*Msol/(365.*24*3600.)
	vinf=Vinf*100000.
	rstar=Rstar*Rsol
	mstar=Mstar*Msol
	ve = np.sqrt(2.*G*mstar/rstar)
	Ve = np.sqrt(2.*G*mstar/rstar)/100000.
	rhowstar = mdot/(4.*np.pi*rstar**2*vinf)
	rhocstar = rhowstar*vinf/ve

	#Some scalling relations. see Owocki 2016
	chiinf = 0.034*(vinf/10.**8)**4*(rstar/10**12)/(Mdot/10**(-6))
	Tinf = 14*10**6*(vinf/10.**8)**2

	#Computing the Alfven radius and closure radius
	Beq=Bd/2.
	eta=(Beq)**2*rstar**2/(mdot*vinf)
	RA=0.3+(eta+0.25)**(0.25)
	Rc = RA # This can be changed occording to the user

	#ADM output (the magnetosphere density components)
	admOUT=admCAL(Nx, Ny, Nz, RA, Rc, Teff, Tinf, chiinf, delta )
	Rhow=admOUT[0]
	Rhoh=admOUT[1]
	Rhoc=admOUT[2]
	rhoh=Rhoh*rhowstar
	rhoc=Rhoc*rhocstar
	rhow=Rhow*rhowstar

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)
	dX=np.abs(XX[0]-XX[1])
	dY=np.abs(YY[0]-YY[1])
	dZ=np.abs(ZZ[0]-ZZ[1])
	dx=dX*Rstar*Rsol
	dy=dY*Rstar*Rsol
	dz=dZ*Rstar*Rsol

	#Creating 3D meshgrids
	Z_grid, X_grid, Y_grid = np.meshgrid( ZZ, XX, YY, indexing='xy')
	R_grid = np.sqrt( Z_grid**2 + X_grid**2 + Y_grid**2 )
	P_grid = np.sqrt( Z_grid**2 + Y_grid**2 )
	MU_grid = X_grid/R_grid                 #cos(theta)
	NU_grid = np.sqrt(1. - MU_grid**2)      #sin(theta)
	CSPHI_grid = Z_grid/P_grid
	SNPHI_grid = Y_grid/P_grid
	CS2PHI_grid = CSPHI_grid**2 - SNPHI_grid**2
	SN2PHI_grid = 2.*SNPHI_grid*CSPHI_grid

	#Depolarisation factor (see Fox 1991)
	#D=np.ones([NNz,NNx,NNy])  #For a point light source
	D=np.sqrt(1.-1./R_grid**2) #For a finite star


	#Variable setup
	Q=np.zeros(PH) #Stokes U
	U=np.zeros(PH) #Stokes U
	for ph in range(PH):

		#Trick for removing occulted regions (probably not the most efficient way )
		RHO=rhoh+rhoc+rhow	
		RHO_rot = np.zeros([NNz,NNx,NNy])
		alpha=np.arccos(csalpha(phi[ph]*2.*np.pi,np.radians(inc),np.radians(beta)))
		for k in range(0,NNx):
			RHO_rot[:,k,:] = rotate(RHO[:,k,:],np.degrees(alpha),reshape=False)
		RHO_rot[ R_grid<1.0 ] = 0.
		RHO_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0 
		for k in range(0,NNx):
			RHO[:,k,:] = rotate(RHO_rot[:,k,:],np.degrees(-alpha),reshape=False)
			
		#Removing data inside star and outside closure radius
		RHO[ R_grid < 1.0 ] = 0. 
		RHO[ R_grid > Rc ] = 0.
		MU_grid[ R_grid < 1.0 ] = 0.
		MU_grid[ R_grid > Rc ] = 0. 
		NU_grid[ R_grid < 1.0 ] = 0.
		NU_grid[ R_grid > Rc ] = 0. 
		D[ R_grid < 1.0 ] = 0.
		D[ R_grid > Rc ] = 0.

		#Electron density
		ne=RHO*alphae/mp

		#Computation of integral moments
		tau0 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma0 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*MU_grid**2,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma1 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*2*MU_grid*NU_grid*CSPHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma2 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*2*MU_grid*NU_grid*SNPHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma3 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*NU_grid**2*CS2PHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma4 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*NU_grid**2*SN2PHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)

		#Calculation of Stokes Q and U curves
		amp = (tau0-3.*taugamma0)*100.
		taugamma1 = taugamma1*100.
		taugamma2 = taugamma2*100.
		taugamma3 = taugamma3*100.
		taugamma4 = taugamma4*100.
		QU = QU_(phi[ph], amp, taugamma1, taugamma2, taugamma3, taugamma4, inc, beta, thetaIS, QIS, UIS)
		Q[ph] = QU[0] 
		U[ph] = QU[1]

	return [Q,U]


def QU_(ph,amp,taugamma1,taugamma2,taugamma3,taugamma4,inc,beta,thetaIS,QIS,UIS):
	inc = np.radians(inc)
	beta = np.radians(beta)
	lda = 2.*np.pi*ph

	V = amp
	W = taugamma1
	X = taugamma2
	Y = taugamma3
	Z = taugamma4

	E = np.sin(2.*inc)*np.cos(thetaIS)
	F = 2.*np.sin(inc)*np.sin(thetaIS)
	J = np.sin(2.*inc)*np.sin(thetaIS)
	K = 2.*np.sin(inc)*np.cos(thetaIS)
	L = (1.+np.cos(inc)**2)*np.cos(thetaIS)
	M = 2.*np.cos(inc)*np.sin(thetaIS)
	N = (1.+np.cos(inc)**2)*np.sin(thetaIS)
	R = 2.*np.cos(inc)*np.cos(thetaIS)

	q0 = V*np.sin(inc)**2*(3.*np.cos(beta)**2 - 1.)*np.cos(thetaIS) + 3.*X*np.sin(inc)**2*np.sin(2.*beta)*np.cos(thetaIS) + 3.*Y*np.sin(inc)**2*np.sin(2.*beta)*np.cos(thetaIS) + QIS
	q1 = 2.*W*E*np.cos(beta) - 2.*Z*E*np.sin(beta) + V*F*np.sin(2.*beta) - 2.*X*F*np.cos(2.*beta) - Y*F*np.sin(2.*beta)
	q2 = V*E*np.sin(2.*beta) - 2.*X*E*np.cos(2.*beta) - Y*E*np.sin(2.*beta) - 2.*W*F*np.cos(beta) - 2.*Z*F*np.sin(beta)
	q3 = -V*L*np.sin(beta)**2 + X*L*np.sin(2.*beta) - Y*L*(1.+np.cos(beta)**2) + 2.*W*M*np.sin(beta) - 2.*Z*M*np.cos(beta)
	q4 = 2.*W*L*np.sin(beta) + 2.*Z*L*np.cos(beta) + V*M*np.sin(beta)**2 - X*M*np.sin(2.*beta) + Y*M*(1+np.cos(beta)**2)
	Q = q0 + q1*np.cos(lda) + q2*np.sin(lda) + q3*np.cos(2.*lda) + q4*np.sin(2.*lda) 

	u0 = V*np.sin(inc)**2*(3.*np.cos(beta)**2 - 1.)*np.sin(thetaIS) + 3.*X*np.sin(inc)**2*np.sin(2.*beta)*np.sin(thetaIS) + 3.*Y*np.sin(inc)**2*np.sin(2.*beta)*np.sin(thetaIS) + UIS
	u1 = 2.*W*J*np.cos(beta) - 2.*Z*J*np.sin(beta) - V*K*np.sin(2.*beta) + 2.*X*K*np.cos(2.*beta) + Y*K*np.sin(2.*beta)
	u2 = V*J*np.sin(2.*beta) - 2.*X*J*np.cos(2.*beta) - Y*J*np.sin(2.*beta) + 2.*W*K*np.cos(beta) + 2.*Z*K*np.sin(beta)
	u3 = -V*N*np.sin(beta)**2 + X*N*np.sin(2.*beta) - Y*N*(1.+np.cos(beta)**2) - 2.*W*R*np.sin(beta) + 2.*Z*R*np.cos(beta)
	u4 = 2.*W*N*np.sin(beta) + 2.*Z*N*np.cos(beta) - V*R*np.sin(beta)**2 + X*R*np.sin(2.*beta) - Y*R*(1+np.cos(beta)**2)
	U = u0 + u1*np.cos(lda) + u2*np.sin(lda) + u3*np.cos(2.*lda) + u4*np.sin(2.*lda) 

	return [Q,U]
