import numpy as np
from numpy.linalg import norm as norm
import scipy as scp
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
np.set_printoptions(precision=3,suppress= True)

class Conditions:
    
    def __init__(self,which = 'PWR'):
        
        self.g = 9.81 # m/s2
        
        if which == 'PWR':
            self.G = 4000 #kg/m2 s
            self.qp0 = 430e2 #W/cm to W/m
            self.P0 = 15e6 #MPa to pa
            self.Tf0c = 277 #oC
        
        else:
            self.G = 2350 #kg/m2 s
            self.qp0 = 605e2 #W/cm to W/m
            self.P0 = 7.5e6 #MPa to pa
            self.Tf0c = 272 #oC
            
        self.Tf0 = self.Tf0c + 273.15

class PinProperties:
    
    def __init__(self,Conditions,which = 'PWR'):
    
        if which == 'PWR':
            
            self.H = 4 #m
            self.Drod = 0.95e-2 # cm to m 
            self.Pitch = 1.26e-2 # cm to m
            self.DFuel = 0.82e-2 # cm to m
            self.Gap_thickness = 0.006e-2 #cm to m
            self.kgap=0.25 #W/mK
            self.k_fuel=3.6 #W/mK
            self.k_cladding=21.5 #W/mK
        
        else:
            
            self.H = 4.1 #m
            self.Drod = 1.227e-2 # cm to m
            self.Pitch = 1.62e-2 # cm to m 
            self.DFuel = 1.04e-2 # cm to m 
            self.Gap_thickness = 0.010e-2 # cm to m
            self.kgap=0.25 #W/mK
            self.k_fuel=3.6 #W/mK
            self.k_cladding=21.5 #W/mK

        
        self.xih = self.Drod * np.pi #m
        
        self.qp = lambda z: Conditions.qp0* np.sin(np.pi * z / self.H) #W/m
        self.qpp = lambda z: self.qp(z) / self.xih # w /m2
        
        return

class FluidProperties:
    
    def __init__(self,ICs,Pin,poly_order=15):
        
        '''
        Steam Table
        -----------
        '''
        props = XSteam(XSteam.UNIT_SYSTEM_BARE)
        
        '''
        Constant Material Properties 
        ----------------------------
        '''
        _p0,_tf0 = ICs.P0/1e6,ICs.Tf0 # Initial Pressure [MPa] and Temp [K]
        self.cp = props.Cp_pt(_p0,_tf0) * 1e3 #kj/kg -> j/kg
        self.mu = props.my_pt(_p0,_tf0)  # N s / m2
        self.k = props.tc_pt(_p0,_tf0)  # W/ m K

        '''
        Pressure Variant Properties
        ---------------------------
        '''
        self.hg = lambda P: props.hV_p(P) * 1e3 # kJ/kg -> j/kg
        self.hf = lambda P: props.hL_p(P) * 1e3 # kJ/kg -> j/kg
        self.hfg = lambda P: self.hg(P) - self.hf(P)
        self.tsat = lambda P: props.tsat_p(P) # K

        '''
        Initial Properties
        ------------------
        '''
        self.X_e0 = self.cp*(_tf0 - self.tsat(_p0))/self.hfg(_p0)
        self.rho0 = props.rho_pt(_p0,_tf0) #kg /m3

        '''
        Pipe Dimensions
        ---------------
        '''
        self.Area = (Pin.Pitch**2 - 1/4 * np.pi * Pin.Drod**2) #m2
        self.xiw = Pin.xih #m
        self.Dh = 4 * self.Area / (self.xiw) #m
        
        '''
        Dimensionless Groups +
        ----------------------
        '''
        self.Re = ICs.G * self.Dh / self.mu
        self.Pr = self.cp * self.mu / self.k
        self.Nu = 0.023 * self.Re**(.8) * self.Pr ** (0.4)
        self.h = self.Nu * self.k / self.Dh 
        self.__f = self.Re ** (-.25) * 0.316

        '''
        Two-Phase Properties
        --------------------
        '''
        self.rhosatratio = lambda P: props.rhoV_p(P) / props.rhoL_p(P)
        self.alpha = lambda Xe, P: 1 / (1 + (Xe**(-1) - 1) * self.rhosatratio(P))
        self.rhofg = props.rhoL_p(_p0) - props.rhoV_p(_p0)
        self.rhoL = props.rhoL_p(_p0)
        self.rhoV = props.rhoV_p(_p0)
        self.rhom = lambda Xe, P: self.rhoL if Xe<0 else (1/self.rhoL + 1/self.rhofg*Xe)**(-1)
            #for props.my_pt, for whatever reason pyXSteam doesnt have mu at saturation
            #so you have to alter tsat a littlebit
        self.__muL = lambda P: props.my_ph(P, props.hL_p(P))
        self.__muV = lambda P: props.my_ph(P, props.hV_p(P))
        self.mum = lambda Xe, P: self.mu if Xe <0 else (Xe / self.__muV(P) + (1 - Xe)/self.__muL(P))**(-1)
        self.f = lambda Xe, P: self.__f if Xe <0 else self.__f* (self.mum(Xe, P) / self.__muL(P))


        '''
        Material Property Fits
        ----------------------
        '''
        self.__pressures = np.linspace(1,20,1000)
        self.__hf_data = [self.hf(P) for P in self.__pressures] 
        self.__hg_data = [self.hg(P) for P in self.__pressures]

        #fitting with 15th order polynomial with numpy.polyfit and np.polynomial.Polynomials
        #polyfit returns high -> low, polynomial.Polynomial takes low -> high 
        #only setting as object for plotting
        self.__hf_fit = np.polynomial.Polynomial( np.flip( np.polyfit(self.__pressures*1e6,self.__hf_data,poly_order) ) ) 
        self.__hg_fit = np.polynomial.Polynomial( np.flip( np.polyfit(self.__pressures*1e6,self.__hg_data,poly_order) ) ) 

        self.dhfdp = self.__hf_fit.deriv()
        self.dhgdp = self.__hg_fit.deriv()
        
        return

    def plot(self):
        
        p,hfdata,hgdata = self.__pressures, self.__hf_data, self.__hg_data
        hf_fit = self.__hf_fit(p*1e6)
        hg_fit = self.__hg_fit(p*1e6)

        hf_norm = norm(hf_fit - hfdata,2) / norm(hfdata,2)
        hg_norm = norm(hg_fit - hgdata,2) / norm(hgdata,2)
        
        fig,ax = plt.subplot_mosaic([['h_f','h_fg']],figsize = (12,5),gridspec_kw={'wspace':.2})
        
        ax['h_f'].plot(p, hfdata, label = 'pyXSteam Data')
        ax['h_f'].plot(p, hf_fit, label = 'h$_f$ Polynomial Fit')
        ax['h_f'].set_ylabel('h$_f$  [J $\cdot$ kg$^{-1}$]')

        ax['h_fg'].plot(p, hgdata, label = 'pyXSteam Data')
        ax['h_fg'].plot(p, hg_fit, label= 'h$_{fg}$ Polynomial Fit')
        ax['h_fg'].set_ylabel('h$_{fg}$  [J $\cdot$ kg$^{-1}$]')

        for plot in ax:
            ax[plot].set_xlabel('Pressure  [MPa]')
            ax[plot].grid()
            ax[plot].legend()

        print(f'hf Fit L2 Norm: {hf_norm}\nhg Fit L2 Norm: {hg_norm}')
        
        return 
        
'''
SOLVER
'''
class Solver:
    
    def __init__(self,which = 'PWR',num_zsteps = 100):
    
        self.ICs = Conditions(which=which)
        self.pin = PinProperties(which=which,Conditions=self.ICs)
        self.fluid = FluidProperties(self.ICs,self.pin)
        self.z = np.linspace(0,self.pin.H,num_zsteps)
        
        return

    def plotSolutions(self):

        self.fluid.plot()
        plt.show()
        
        fig,ax = plt.subplots()
        ax.plot(self.XeFunc(self.z), self.z)
        ax.set_xlabel('Equilibrium Steam Quality')
        ax.set_ylabel('Axial Position  [m]')
        ax.grid()
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(self.TfFunc(self.z),self.z, label = 'Mean Fluid Temperature')
        tsat = [self.fluid.tsat(p * 1e-6) for p in self.PFunc(self.z)]
        ax.plot(tsat,self.z, label = 'Saturation Temperature')
        ax.set_xlabel('Temperature  [K]')
        ax.set_ylabel('Axial Position  [m]')
        ax.legend()
        ax.grid()
        plt.show()

    def solveFluid(self,output = True):
        
        ICs,pin, fluid = self.ICs, self.pin, self.fluid
        z = self.z
        P = [ICs.P0]
        X_e = [fluid.X_e0]
        T_f = [ICs.Tf0]
        if output:
            print('Z-Coordinate\tPressure\tSteam Quality\tFluid Temperature')
            print('-----------------------------------------------------------------')
            print(f'{np.trunc(z[0],)}\t\t{np.around(P[0]*1e-6,3)}\t\t{np.around(X_e[0],4)}\t\t{np.around(T_f[0],3)}')
        for i in range(1, len(z)):
            
            dz = z[i]- z[i-1]
            
            '''
            Solving for Pressure
            Solve for pressure using the material properties at the previous z
            '''
            #helperP = (fluid.f(X_e[i-1],P[i-1]*1e-6)* ICs.G**2* fluid.xiw)/ (2* fluid.rhom(X_e[i-1],P[i-1]*1e-6)* fluid.Area)
            #P_i = -dz * (helperP  + fluid.rhom(X_e[i-1],P[i-1]*1e-6)* ICs.g) + P[i-1]
            rhom = fluid.rhom(X_e[i-1],P[i-1]*1e-6)
            topP1 = 1/2 * fluid.xiw / fluid.Area * fluid.f(X_e[i-1],P[i-1]*1e-6) * ICs.G**2 / rhom
            topP2 = rhom * ICs.g
            topP3 = ICs.G * pin.qp(z[i]) / (fluid.rhofg * fluid.Area * fluid.hfg(P[i-1]*1e-6))
            botP1 =  0 if X_e[i-1] < 0 else X_e[i-1] * fluid.dhgdp(P[i-1]) + (1-X_e[i-1]) * fluid.dhfdp(P[i-1])
            botP2 = 1 - ICs.G**2 / (fluid.rhofg * fluid.hfg(P[i-1]*1e-6)) * botP1
            insideP = (topP1 + topP2 + topP3) / botP2
            P_i = P[i-1] - dz *insideP

            '''
            Solving for Steam Quality
            Solve for steam quality using current pressure
            '''
            dpdz = (P_i - P[i-1]) / dz
            insideXe1 = fluid.dhfdp(P_i) * dpdz if X_e[i-1] < 0 else X_e[i-1] * fluid.dhgdp(P_i) * dpdz + (1 - X_e[i-1]) * fluid.dhfdp(P_i) * dpdz
            insideXe2 = pin.qp(z[i]) / (fluid.Area * ICs.G * fluid.hfg(P_i* 1e-6)) - 1/fluid.hfg(P_i * 1e-6) * insideXe1
            X_e_i = dz * insideXe2 + X_e[i-1]

            '''
            Solving for Temperature
            Find temperature with current properties
            '''
            T_f_i = fluid.hfg(P_i*1e-6) / fluid.cp * X_e_i + fluid.tsat(P_i*1e-6)

            '''
            Printing
            '''
            if output:
                print(f'{np.around(z[i],3)}\t\t{np.around(P_i*1e-6,3)}\t\t{np.around(X_e_i,4)}\t\t{np.around(T_f_i,3)}')
            
            '''
            Appending Pressure, Steam Quality, and Fluid Temperature lists
            lists are better than numpy arrays for on the fly changing...
            '''
            P.append(P_i)
            X_e.append(X_e_i)
            T_f.append(T_f_i)
            
        
        self.PFunc = scp.interpolate.interp1d(z,P)
        self.XeFunc = scp.interpolate.interp1d(z,X_e)
        self.TfFunc = scp.interpolate.interp1d(z,T_f)
        
        return


    def solveCladSurface(self):

        T_cs = lambda z: self.__pin.qpp(z) / self.__fluid.h + self.TfluidFunc(z)

        rho_ratio = lambda z: self.__fluid.props.rhoL_p(self.Pfunc(z)*1e-6) / self.__fluid.props.rhoV_p(self.Pfunc(z)*1e-6)

        F = lambda z: (1+self.Xefunc(z)  * self.__fluid.Pr *(rho_ratio(z) - 1))**(.35)
        S = lambda z: 1 / (1 + 0.055 * F(z)**(.1) * self.__fluid.Re**(.16))

        self.T_cs = T_cs

        return

    def solvePinTemperature(self):
    
        pin = self.__pin
        fluid = self.__fluid
        Rf = pin.DFuel /2
        Rci = pin.DFuel /2 + pin.Gap_thickness
        Rco = pin.Drod / 2
        Aci = 2*Rci * np.pi
        Af = 2*Rf * np.pi
        Axf = Rf**2 * np.pi
        
        C2 = lambda z: pin.qp(z) * Rf * Rci * Af/ pin.k_cladding / Aci / Axf
        C3 = lambda z: self.T_cs(z) - C2(z)*np.log(Rco)
        T_clad = lambda r,z: C2(z)*np.log(r) + C3(z)

        C1 = lambda z: (pin.qpp(z) + fluid.h * Aci * T_clad(Rci,z) ) / fluid.h / Af  -  pin.qp(z)*Rf**2 / (4 * Axf * pin.k_fuel)
        T_fuel = lambda r, z: -pin.qp(z) * r**2 / 4 / pin.k_fuel / Axf + C1(z)

        self.T_clad = T_clad
        self.T_fuel = T_fuel
        return
        













