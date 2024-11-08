import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
np.set_printoptions(precision=5,suppress= True)

class Properties:
    def __init__(self,Conditions,Geometry):
        
        props = XSteam(XSteam.UNIT_SYSTEM_BARE)
        self.props = props
        _p0,_tf0 = Conditions.P0/1e6,Conditions.Tf0
        
        self.hfg = lambda P: (props.hV_p(P) - props.hL_p(P)) * 1e3
        self.hf = lambda P: props.hL_p(P) * 1e3
        self.tsat = lambda P: props.tsat_p(P)
        
        self.cp = props.Cp_pt(_p0,_tf0) * 1e3 #j/kg
        self.mu = props.my_pt(_p0,_tf0)  # N s / m2
        self.k = props.tc_pt(_p0,_tf0)  #
        self.rho = props.rho_pt(_p0,_tf0)

        self.X_e0 = self.cp*(_tf0 - self.tsat(_p0)) / self.hfg(_p0)
        
        self.Area = (Geometry.Pitch**2 - 1/4 * np.pi * Geometry.Drod**2)
        self.xiw = Geometry.xih
        self.Dh = 4 * self.Area / (self.xiw)
        
        self.Re = Conditions.G * self.Dh / self.mu
        self.Pr = self.cp * self.mu / self.k
        self.Nu = 0.023 * self.Re**(.8) * self.Pr ** (0.4)
        self.h = self.Nu * self.k / self.Dh
        self.f = self.Re ** (-.25) * 0.316
        
        def fit(x, A,B,C,D,E,F,G,H):
            return A + B*x + C*x**2 + D * x**3 + E* x**4 + F*x**5 + G*x**6 + H*x**7

        self.__pressures = np.linspace(1,20,1000)
        _hf = [self.hf(_p) for _p in self.__pressures]
        _hfg = [self.hfg(_p) for _p in self.__pressures]
        self.hf_fit = np.polynomial.Polynomial(scp.optimize.curve_fit(fit,self.__pressures*1e6,_hf)[0]) 
        self.hfg_fit = np.polynomial.Polynomial(scp.optimize.curve_fit(fit,self.__pressures*1e6,_hfg)[0]) 
        self.dhfdp = self.hf_fit.deriv()
        self.dhfgdp = self.hfg_fit.deriv()
        return
        
    def plot_fit(self,fit='hfg'):
        
        if fit == 'hfg':
            print(f'fit: {self.hfg_fit}')
            hfg = [self.hfg(_p) for _p in self.__pressures]
            plt.plot(self.__pressures,hfg,label = 'Real Data')
            plt.plot(self.__pressures,self.hfg_fit(self.__pressures),label = 'Data Fit')
            
        elif fit =='hf':
            print(f'fit: {self.hf_fit}')
            hf = [self.hf(_p) for _p in self.__pressures]
            plt.plot(self.__pressures,hf, label = 'Real Data')
            plt.plot(self.__pressures,self.hf_fit(self.__pressures), label = 'Data Fit')

        plt.legend()
        plt.grid()
        plt.xlabel('Pressure  [MPa]')
        plt.ylabel(f'{fit}  []')


class Geometry:
    
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
            self.P0 = 7.5e6 #MPa
            self.Tf0c = 272 #oC
            
        self.Tf0 = self.Tf0c + 273.15
        




