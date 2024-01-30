#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#import os
#os.environ['OPENBLAS_NUM_THREADS']='1'
from numpy import *
#from pylab import*
from numpy.fft import fftfreq , fft , ifft , irfft2 , rfft2
from mpi4py import MPI
import time

start_time = time.time()
nu         = 10**-30
T          = 0.2 
dt         = 0.01
t          = 0.0
tstep      = 0
t_snap     = 1
N          = 360 
N2         = N ;
alpha      = 20 
Ro         = 0.1

comm          = MPI. COMM_WORLD
num_processes = comm. Get_size ()
rank          = comm. Get_rank ()
Np            = int(N / num_processes)

# X cordinates
X             = mgrid [rank*Np :( rank+1)*Np , :N, :N]. astype (float )*2*pi/N


# Velocities
U             = empty ((3,Np,N,N))
U_W           = empty ((3,Np,N,N))
U_G           = empty ((3,Np,N,N))
U_W_hat       = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
U_hat         = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
U_hat_ini     = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
U_G_hat       = empty ((3, N, Np ,int(N/2)+1), dtype = complex );
U_hat0        = empty ((3, N, Np , int(N/2)+1), dtype = complex )
U_hat1        = empty ((3, N, Np , int(N/2)+1), dtype = complex )
Uc_hat        = empty ((N, Np , int(N/2)+1), dtype = complex )
Uc_hatT       = empty ((Np , N, int(N/2)+1), dtype = complex )
U_mpi         = empty (( num_processes , Np , Np , int(N/2)+1), dtype = complex )


#Buoyancy
B             = empty ((3,Np,N,N))
B_W           = empty ((3,Np,N,N))
B_G           = empty ((3,Np,N,N))
B_hat         = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
B_hat_ini     = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
B_W_hat       = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
B_G_hat       = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
B_hat0        = empty ((3, N, Np ,int(N/2)+1), dtype = complex )
B_hat1        = empty ((3, N, Np ,int(N/2)+1), dtype = complex )

#Pressure
P             = empty ((Np , N, N))
P_hat         = empty ((N, Np , int(N/2)+1), dtype = complex )
P_W           = empty ((Np , N, N))
P_W_hat       = empty ((N, Np , int(N/2)+1), dtype = complex )
P_G           = empty ((Np , N, N))
P_G_hat       = empty ((N, Np , int(N/2)+1), dtype = complex )


#Non-linear terms
terms_Ro      = empty ((3,Np,N,N))
terms_Ro_hat  = empty ((3, N, Np ,int(N/2)+1), dtype = complex ) 

#Right hand side of govn Eqn
dU            = empty ((3, N, Np , int(N/2)+1), dtype = complex )
dB            = empty ((3, N, Np , int(N/2)+1), dtype = complex )

# Wavenumbers
kx            = fftfreq (N, 1./N)
ky            = kx[rank*Np :( rank+1)*Np]
kz            = kx [:(int(N/2)+1)]. copy ();kz[-1] *= -1; 
kz1           = kz ;
kz1[0]        = 1 ;
K_Sp          = array ( meshgrid (kx , ky, kz1 , indexing ='ij'), dtype =int) 
K             = array ( meshgrid (kx , ky, kz , indexing ='ij'), dtype =int)

K1            = array ( meshgrid (kx , ky, alpha*kz , indexing ='ij'), dtype =int)
K2            = sum(K1*K1,0,dtype=int)
K_lap         = sum(K*K,0,dtype=float)
K_over_K2     = K. astype (float ) / where (K2 == 0, 1, K2). astype (float)
kmax_dealias  = 2./3.*(N/2+1)
dealias       = array (( abs(K[0]) < kmax_dealias )*( abs(K[1]) < kmax_dealias )*(abs(K[2]) < kmax_dealias ), dtype =bool)
#dealias       =  exp(-36*(abs(K[0])/N)**36) * exp(-36*(abs(K[1])/N)**36) * exp(-36*(abs(K[2])/N)**36);
a             = [1./6., 1./3., 1./3., 1./6.]
b             = [0.5, 0.5, 1.]


# The hyper dissipation

M             = ones (( N, Np ,int(N/2)+1), dtype = complex );
M_dissip_per  = nu*(K_lap)**8;
Mv            = M+dt*M_dissip_per ;

# The Fourier and Inverse Transforms

def ifftn_mpi (fu , u):
    Uc_hat [:]   = ifft(fu , axis=0)
    comm. Alltoall ([ Uc_hat , MPI. DOUBLE_COMPLEX ], [U_mpi , MPI. DOUBLE_COMPLEX ])
    Uc_hatT [:]  = rollaxis (U_mpi , 1). reshape ( Uc_hatT . shape)
    u[:] = irfft2 (Uc_hatT , axes =(1, 2))
    return u
def fftn_mpi (u, fu):
    Uc_hatT [:] = rfft2 (u, axes =(1,2))
    U_mpi [:] = rollaxis ( Uc_hatT . reshape (Np , num_processes , Np , int(N/2)+1), 1)
    comm. Alltoall ([ U_mpi , MPI. DOUBLE_COMPLEX ], [fu , MPI. DOUBLE_COMPLEX ])
    fu [:] = fft(fu , axis=0)
    return fu

# The derivatives in the three directions

def diff_x(a):
    b = empty ((3,Np,N,N));
    b[0] = ifftn_mpi(1j*K[0]*a,b[0]);
    return b[0];
def diff_y(a):
    b = empty ((3,Np,N,N));
    b[0] = ifftn_mpi(1j*K[1]*a,b[0]);
    return b[0];
def diff_z(a):
    b = empty ((3,Np,N,N));
    b[0] = ifftn_mpi(1j*K[2]*a,b[0]);
    return b[0];

# Compute RHS of the Bouyancy equation

def computeRHSdB (dB,rk):
    Ro_terms    = empty ((3, N, Np , int(N/2)+1), dtype = complex )
    dB[2]       = -U_hat[2]
    if rk > 0:
        for i in range(3):
           U[i]            = ifftn_mpi ( U_hat [i], U[i]);
    Ro_terms[2] = fftn_mpi(-Ro*(U[0]*diff_x(B_hat[2])+U[1]*diff_y(B_hat[2])+U[2]*diff_z(B_hat[2])),Ro_terms[2])
    dB[2]      += Ro_terms[2];
    dB[1]       = 0
    dB[0]       = 0
    return dB 


#Compute RHS of the velocity equations

def computeRHS (dU,rk):
    dU[0] = U_hat[1]; dU[1] = -U_hat[0] ; dU[2] = alpha**2 * B_hat[2];
    
    if rk > 0:
        for i in range(3):
           U[i]            = ifftn_mpi ( U_hat [i], U[i]);
    for i in range(3):
        terms_Ro_hat[i] = +Ro*fftn_mpi((U[0]*diff_x(U_hat[i])+U[1]*diff_y(U_hat[i])+U[2]*diff_z(U_hat[i])),terms_Ro_hat[i])
        dU[i]          -= terms_Ro_hat[i];
    P_hat[:]        = (sum(K_over_K2*dU,0,out=P_hat))#Div(dU,P_hat) 
    dU[0]          -= P_hat*K[0] ;
    dU[1]          -= P_hat*K[1] ;
    dU[2]          -= P_hat*alpha**2*K[2] ;
    return dU


# Function to save data

def writeData(ufull,vfull,wfull,bfull,t,rank):
    savez('t_%d/U_%d.npz'%(t,rank),ufull=ufull, vfull=vfull, wfull=wfull, bfull=bfull);
    return(0);

# Summing over

def sum_over_numproc(quantity):
    localsum=sum(quantity)
    # sending data from each processor to root processor
    if rank!=0:
       destination_process=0
       comm.send(localsum, dest=destination_process )

    if rank==0:
       globalsum=localsum
       for rnk in range(1,num_processes):
           globalsum=globalsum+comm.recv(source=rnk)
       for rnk in range(1,num_processes):
           comm.send(globalsum, dest=rnk )

    if rank!=0:
       globalsum=comm.recv(source=0)

    return(globalsum)

def Evortexwave(uw,vw,bw,ug,vg,bg,u0,v0,w0,b0):     
    dvol              = (2.0*pi/N)**2*(pi/N2)
    alpha             =  20
    WaveEnergy        =  sum_over_numproc(0.5*(uw*uw+vw*vw+w0*w0/alpha**2+bw*bw))*dvol
    VortexEnergy      =  sum_over_numproc(0.5*(ug*ug+vg*vg+ bg*bg))*dvol
    TotalEnergy       =  sum_over_numproc(0.5*(u0*u0+v0*v0+ b0*b0+w0*w0/alpha**2))*dvol
    return WaveEnergy,VortexEnergy,TotalEnergy

# Wave Initialisation

def InitializeWaves(u0,v0,w0,b0):

   # Create random pressure distribution in Fourier Space
   #amp_initial = 1.1109481e-2   # multiply by a factor of 10 to increase energy 100 folds. 
   #seed(121)
   rand_theta     =  rand(600)*2.0*pi #for declaring in theta space

   #seed(233)
   rand_z_amp     =  (rand(600)-0.5)
   
   # Create random pressure distribution in Fourier Space
   amp_initial    =  0.0018947861439011286*sqrt(2)*10**3
   K_sqre         =  K[0]**2+K[1]**2
   K_sqre[0][0]   =  1 ;
   P0             =  zeros((Np,N,N2))
   ctr            =  0;  ctrz  = 0;
   k_init         =  7
   for i in range(1,k_init):
       for j in range(1,k_init):
           for k in range(1,k_init):
               P0 = P0 +(i**2+j**2+k**2)**-7*( amp_initial*sin(float(i)*X[0]+rand_theta[ctr])*sin(float(j)*X[1]+rand_theta[ctr+1])*cos(float(k)*X[2])*rand_z_amp[ctrz])
               ctrz=ctrz+1
           ctr=ctr+2
   
   P_W_hat[:] = fftn_mpi(P0,P_W_hat)
   Pshell     = (K[0]*K[0]+K[1]*K[1]<=(k_init-1)**2)*(K[2]<=k_init-1)
   Pshell1    = (K[0]*K[0]>0 )*(K[1]*K[1]>0)*( K[2]> 0 )
   P_W_hat[:] = P_W_hat*Pshell*Pshell1 ;
   sigma_sqre =  alpha**2*(K[2]**2+K_sqre)/(alpha**2*K[2]**2+K_sqre);
   sigma      =  sqrt(sigma_sqre)
   U_W_hat[0] = (P_W_hat/(sigma_sqre-1))*(1j*K[1]+sigma*K[0]);
   U_W_hat[1] = (P_W_hat/(sigma_sqre-1))*(-1j*K[0]+sigma*K[1]);
   U_W_hat[2] = -sigma*K_sqre*P_W_hat/(K_Sp[2]*(sigma_sqre-1)) ;
   B_W_hat[0] = 0 ;
   B_W_hat[1] = 0 ;
   B_W_hat[2] = (1j*(K_sqre)*P_W_hat)/(K_Sp[2]*(sigma_sqre-1))
   
   b0         = ifftn_mpi(B_W_hat[2],b0);
   u0         = ifftn_mpi(U_W_hat[0],u0);
   v0         = ifftn_mpi(U_W_hat[1],v0);
   w0         = ifftn_mpi(U_W_hat[2],w0);
   
   return(u0,v0,w0,b0)

# Balanced flow initialisation

def InitializeVortex(u0,v0,w0,b0):

   # Quasi-geostropic initialization  
   #seed(151)
   rand_theta       =  rand(600)*2.0*pi #for declaring in theta space

   #seed(203)
   rand_z_amp       =  rand(600)-0.5
   
   # Create random pressure distribution in Fourier Space

   amp_initial      =  0.0018947861439011286*sqrt(0.06)*10
   k_init           =  7 
   P0               =  zeros((Np,N,N2))
   ctr              =  0  ;   ctrz = 0;
   for i in range(1,k_init):
       for j in range(1,k_init):
           for k in range(1,k_init):
               P0 = P0 + (i**2+j**2+k**2)**-7*(amp_initial*sin(float(i)*X[0]+rand_theta[ctr])*sin(float(j)*X[1]+rand_theta[ctr+1])*cos(float(k)*X[2])*rand_z_amp[ctrz])
               ctrz=ctrz+1
           ctr=ctr+2

   P_G_hat[:] = fftn_mpi(P0,P_G_hat)
   
   # Remove zero kh modes (shear) , also eliminate high wavenumber modes
   #Pshell = ( abs(KX)<6.0 )*( abs(KY)<6.0 )*( abs(KC)<= 6.0 ) 
   
   Pshell       = (K[0]*K[0]+K[1]*K[1]<=(k_init-1)**2)*(K[2]<=k_init-1)
   Pshell1      = (K[0]*K[0]>0 )*(K[1]*K[1]>0)*( K[2]> 0 )


   P_G_hat[:]   = P_G_hat*Pshell*Pshell1

   P_G[:]       = ifftn_mpi(P_G_hat,P_G);
   u0           = u0 - diff_y(P_G_hat);
   v0           = v0 + diff_x(P_G_hat);
   w0           = w0 + zeros((Np,N,N2));
   b0           = b0 + diff_z(P_G_hat)

   return(u0,v0,w0,b0)

#Call initialisation

#[U_W[0],U_W[1],U_W[2],B_W[2]] = InitializeWaves(U_W[0],U_W[1],U_W[2],B_W[2]);
#[U_G[0],U_G[1],U_G[2],B_G[2]] = InitializeVortex(U_G[0],U_G[1],U_G[2],B_G[2]);


#Total velocity
U[0]         =  sin(X[0])*cos(X[1])*cos(X[2])
U[1]         =  -cos(X[0])*sin(X[1])*cos(X[2])
U[2]         =  0
B[0]         =  0 # B_W + B_G ;
B[1]         =  0 
B[2]         =  sin(X[0])*cos(X[1])*sin(X[2]) 
#P         =   P_W + P_G


#Save initial velocity
writeData(U[0],U[1],U[2],B[2],tstep,rank)

# Convert to fourier space

for i in range(3):
    U_hat [i] = fftn_mpi (U[i], U_hat [i]);
    B_hat [i] = fftn_mpi (B[i], B_hat [i]);

if rank==0:print(amax(U_hat));

# Inorder to force

Fshell        =  (K[0]*K[0]+K[1]*K[1]<=4**2)*(K[2]<=4)
Fshell1       =  (K[0]*K[0]>0 )*(K[1]*K[1]>0)*( K[2]> 0 )
U_hat_ini     =  U_hat*Fshell*Fshell1 ;
B_hat_ini     =  B_hat*Fshell*Fshell1 ;
#Theta_hat_ini =  Theta_hat*Fshell*Fshell1
Fshell2       =  (K[0]*K[0]+K[1]*K[1]>4**2)*(K[2]>4)

#Initial E_G and E_W

E_W, E_G, E_T = Evortexwave(U_W[0],U_W[1],B_W[2],U_G[0],U_G[1],B_G[2],U[0],U[1],U[2],B[2])
if rank==0:
    print('E_W is %lf, E_G is %lf and E_T is %lf\n'%(E_W,E_G,E_T));
    file = open('Energies.txt','w');
    file.write('time\tE_W\tE_G\tE_T\n');
    file.write('%12.12f\t%12.12f\t%12.12f\t%12.12f\n'%(t,E_W,E_G,E_T));


# Time evolution
while t < T-1e-8:
    t += dt; tstep += 1
    U_hat1 [:]     = U_hat0 [:]    = U_hat
    B_hat1[:]      = B_hat0[:]     = B_hat 
    for rk in range(4):
        dU      = computeRHS (dU, rk)
        dB      = computeRHSdB(dB,rk)
        if rk < 3: 
            U_hat[:]     = U_hat0 + b[rk ]* dt*dU
            B_hat[:]     = B_hat0 + b[rk ]* dt*dB
        
        U_hat1 [:] += a[rk ]* dt*dU
        B_hat1 [:] += a[rk ]* dt*dB
    
    U_hat[:]     = U_hat1[:]
    B_hat[:]     = B_hat1[:] 
    U_hat[:]     = dealias*U_hat[:]/(Mv);
    B_hat[:]     = dealias*B_hat[:]/(Mv);
    
    U_hat        = U_hat*Fshell2 + U_hat_ini
    B_hat        = B_hat*Fshell2 + B_hat_ini
    #Theta_hat    = Theta_hat*Fshell2 + Theta_hat_ini
    
    for i in range(3):
        U[i] = ifftn_mpi (U_hat [i], U[i])
        B[i] = ifftn_mpi (B_hat[i],B[i]);
        

    # Save data

    if (mod(tstep,t_snap)==0):
       writeData(U[0],U[1],U[2],B[2],tstep,rank)
    
if rank==0:
    file=open('Total_time_taken.txt','w')
    file.write('Total step is %d and time taken is %lf'%(tstep,time.time()-start_time));
       
