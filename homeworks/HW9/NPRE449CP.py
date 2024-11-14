import numpy as np
from numpy.linalg import norm as norm
import scipy as scp
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
np.set_printoptions(precision=5,suppress= True)

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

        
        self.xih = self.Drod * np.pi
        
        self.qp = lambda z: Conditions.qp0* np.sin(np.pi * z / self.H)
        self.qpp = lambda z: self.qp(z) / self.xih
        
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
        self.hfg = lambda P: (props.hV_p(P) - props.hL_p(P)) * 1e3 # kJ/kg -> j/kg
        self.hf = lambda P: props.hL_p(P) * 1e3 # kJ/kg -> j/kg
        self.tsat = lambda P: props.tsat_p(P) # K

        '''
        Initial Properties
        ------------------
        '''
        self.X_e0 = self.cp*(_tf0 - self.tsat(_p0))/self.hfg(_p0)
        self.rho0 = props.rho_pt(_p0,_tf0)

        '''
        Pipe Dimensions
        ---------------
        '''
        self.Area = (Pin.Pitch**2 - 1/4 * np.pi * Pin.Drod**2)
        self.xiw = Pin.xih
        self.Dh = 4 * self.Area / (self.xiw)
        
        '''
        Dimensionless Groups +
        ----------------------
        '''
        self.Re = ICs.G * self.Dh / self.mu
        self.Pr = self.cp * self.mu / self.k
        self.Nu = 0.023 * self.Re**(.8) * self.Pr ** (0.4)
        self.h = self.Nu * self.k / self.Dh 
        self.f = self.Re ** (-.25) * 0.316

        '''
        Material Property Fits
        ----------------------
        '''
        self.__pressures = np.linspace(1,20,1000)
        self.__hf_data = [self.hf(P) for P in self.__pressures] 
        self.__hfg_data = [self.hfg(P) for P in self.__pressures]

        #fitting with 9th order polynomial with numpy.polyfit and np.polynomial.Polynomials
        #polyfit returns high -> low, polynomial.Polynomial takes low -> high 
        #only setting as object for plotting
        self.__hf_fit = np.polynomial.Polynomial( np.flip( np.polyfit(self.__pressures*1e6,self.__hf_data,poly_order) ) ) 
        self.__hfg_fit = np.polynomial.Polynomial( np.flip( np.polyfit(self.__pressures*1e6,self.__hfg_data,poly_order) ) ) 

        self.dhfdp = self.__hf_fit.deriv()
        self.dhfgdp = self.__hfg_fit.deriv()
        
        return

    def plot(self):
        
        p,hfdata,hfgdata = self.__pressures, self.__hf_data, self.__hfg_data
        hf_fit = self.__hf_fit(p*1e6)
        hfg_fit = self.__hfg_fit(p*1e6)

        hf_norm = norm(hf_fit - hfdata,2) / norm(hfdata,2)
        hfg_norm = norm(hfg_fit - hfgdata,2) / norm(hfgdata,2)
        
        fig,ax = plt.subplot_mosaic([['h_f','h_fg']],figsize = (12,5),gridspec_kw={'wspace':.2})
        
        ax['h_f'].plot(p, hfdata, label = 'pyXSteam Data')
        ax['h_f'].plot(p, hf_fit, label = 'h$_f$ Polynomial Fit')
        ax['h_f'].set_ylabel('h$_f$  [J $\cdot$ kg$^{-1}$]')

        ax['h_fg'].plot(p, hfgdata, label = 'pyXSteam Data')
        ax['h_fg'].plot(p, hfg_fit, label= 'h$_{fg}$ Polynomial Fit')
        ax['h_fg'].set_ylabel('h$_{fg}$  [J $\cdot$ kg$^{-1}$]')

        for plot in ax:
            ax[plot].set_xlabel('Pressure  [MPa]')
            ax[plot].grid()
            ax[plot].legend()

        print(f'hf Fit L2 Norm: {hf_norm}\nhfg Fit L2 Norm: {hfg_norm}')
        
        return 
        
'''
SOLVER
'''
class Solver:
    
    def __init__(self,which = 'PWR',num_zsteps = 100):
    
        self.__ICs = Conditions(which=which)
        self.__pin = PinProperties(which=which,Conditions=self.__ICs)
        self.__fluid = FluidProperties(self.__ICs,self.__pin)
        self.z = np.linspace(0,self.__pin.H,num_zsteps)
        
        return

    def plotSolutions(self):

        self.__fluid.plot()
        plt.show()
        
        fig,ax = plt.subplots()
        ax.plot(self.Xefunc(self.z), self.z)
        ax.set_xlabel('Equilibrium Steam Quality')
        ax.set_ylabel('Axial Position  [m]')
        ax.grid()
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(self.TfluidFunc(self.z),self.z, label = 'Mean Fluid Temperature')
        tsat = [self.__fluid.tsat(p * 1e-6) for p in self.Pfunc(self.z)]
        ax.plot(tsat,self.z, label = 'Saturation Temperature')
        ax.set_xlabel('Temperature  [K]')
        ax.set_ylabel('Axial Position  [m]')
        ax.legend()
        ax.grid()
        plt.show()

        
    def solveMomentum(self):
        
        ICs,fluid = self.__ICs, self.__fluid
        z = self.z
        pressure = [ICs.P0]
        
        for i in range(len(z)-1):
            dz = z[i+1]- z[i]
            inside = (fluid.f* ICs.G**2* fluid.xiw)/ (2* fluid.rho0* fluid.Area)+ fluid.rho0* ICs.g
            pressure.append(-dz * inside + pressure[i])
            
        pressure = np.array(pressure)
        
        self.Pfunc = np.polynomial.Polynomial(np.flip(np.polyfit(z,pressure,1)))
        
        return

    def solveEnergy(self):
        
        ICs, pin, fluid = self.__ICs, self.__pin, self.__fluid
        z, Pfunc = self.z,self.Pfunc
        
        X_e = [fluid.X_e0]
        T_f = [ICs.Tf0]
        
        for i in range(len(z)-1):
            dz = z[i+1]- z[i]
            zi = z[i]
            mpa = Pfunc(zi)*1e-6
            '''
            Not super sure why inner_chunk*1e4, then by 1e-4 but it works
            I suspect I messed up an area somewhere
            '''
            inner_chunk = (X_e[i] * fluid.dhfgdp(Pfunc(zi)) + fluid.dhfdp(Pfunc(zi))) * Pfunc.deriv()(zi)*1e4
            next_X_e = dz * 1/ fluid.hfg(mpa) * (pin.qp(zi) / fluid.Area - inner_chunk)*1e-4 + X_e[i]
            X_e.append(next_X_e)
            T_f.append(fluid.hfg(mpa) / fluid.cp * next_X_e + fluid.tsat(mpa))

        self.Xefunc = scp.interpolate.interp1d(z,X_e)
        self.TfluidFunc = scp.interpolate.interp1d(z,T_f)

        return

    def solveCladSurface(self):

        T_cs = lambda z: self.__pin.qpp(z) / self.__fluid.h + self.TfluidFunc(z)

        rho_ratio = lambda z: self.__fluid.props.rhoL_p(self.Pfunc(z)*1e-6) / self.__fluid.props.rhoV_p(self.Pfunc(z)*1e-6)

        F = lambda z: (1+self.Xefunc(z)  * self.__fluid.Pr *(rho_ratio(z) - 1))**(.35)
        S = lambda z: 1 / (1 + 0.055 * F(z)**(.1) * self.__fluid.Re**(.16))

        self.T_cs = T_cs

        return

    def solvePinTemperature(self)

        
        













