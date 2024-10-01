import numpy as np 
import numpy.linalg as la
import pandas as pd
from pyXSteam.XSteam import XSteam as xs
props = xs(xs.UNIT_SYSTEM_BARE) #set units to m/kg/sec/K/MPa/W

'''
Basic Definitions
'''
def wT(hin,houts,eta):
    return eta*(hin-houts)

def wP(hin,houts,eta):
    return (hin-houts)/eta

'''
Question 1
'''
print('\nQuestion 1\n')

'''
Getting Material Properties
'''
t1,t5,t4 = np.array([400,20,82])+273.15
h1 = props.h_pt(3,t1)
h2 = props.h_px(.05,.95)
h3 = props.hL_p(.05)
h4 = props.h_pt(3,t4)
h5 = props.h_pt(0.101325,t5)
h6 = props.h_pt(0.101325,t5+15)

'''
Getting work / q
'''
wt = h1-h2
wp = h4-h3
qsg = h1-h4

eta = (wt-wp)/qsg

mdot_ratio = -(h2-h3)/(h5-h6)
print('Thermal Efficiency: {}\n\nMass Flow Ratio (m1/m2): {}'.format(eta,mdot_ratio))

'''
Question 2
'''
print('\n\n\nQuestion 2\n')

thigh,tlow = 293+273.15,33+273.15
'''
Getting Material Properties
'''
#First cycle

c1_h1 = props.hV_t(thigh)
c1_p1 = props.psat_t(thigh)
c1_h2 = props.h_ps(props.psat_t(tlow),props.sV_t(thigh))
c1_h3 = props.hL_t(tlow)
c1_h3p = props.h_ps(c1_p1,props.sL_t(tlow))
c1_h4 = props.hL_t(thigh)

c1_eta = np.around(((c1_h1-c1_h2)-(c1_h3p-c1_h3))/(c1_h1-c1_h3p),5)
c1_steam = np.around(3600/((c1_h1-c1_h2)-(c1_h3p-c1_h3)),5)

#Second Cycle

c2_h1 = props.hV_t(thigh)
c2_h2 = props.h_ps(props.psat_t(tlow),props.sV_t(thigh))
c2_h4 = props.hL_t(thigh)
c2_h3 = props.h_ps(props.psat_t(tlow), props.sL_t(thigh))

c2_eta = np.around(((c2_h1-c2_h2)-(c2_h4-c2_h3))/(c2_h1-c2_h4),5)
c2_steam = np.around(3600 / ((c2_h1-c2_h2)-(c2_h4-c2_h3)),5)

#Third Cycle

c3_h1 = props.h_pt(5,thigh)
c3_h2 = props.h_ps(props.psat_t(tlow),props.s_pt(5,thigh))
c3_h3 = props.hL_t(tlow)
c3_h3p = props.h_ps(5,props.sL_t(tlow))
c3_h4 = props.hL_t(thigh)
c3_h5 = props.hV_t(thigh)

c3_eta = np.around(((c3_h1-c3_h2) - (c3_h3p-c3_h3))/(c3_h1-c3_h3p),5)
c3_steam = np.around(3600 / ((c3_h1-c3_h2) - (c3_h3p-c3_h3)),5)

df = pd.DataFrame([[c1_eta, c1_steam],[c2_eta,c2_steam],[c3_eta,c3_steam]],
             index=['Cycle 1','Cycle 2', 'Cycle 3'],columns=['Thermal Efficiency','Steam Rate'],)
print(df)

'''
Question 3
'''
print('\n\n\nQuestion 3\n')
'''
Getting Basic enthalpies
'''
etat,etap = .9,.85
h1 = props.hV_p(6.890)
h2 = props.h_ps(1.380,props.sV_p(6.890))
h3 = props.hV_p(1.380)
h4 = props.hL_p(1.380)
h5 = props.h_ps(6.89/1000,props.sV_p(1.380))
h6 = props.hL_p(6.89/1000)
h7 = props.h_ps(1.380,props.sL_p(6.89/1000))

'''
Getting Basic Works
'''
wt1 = wT(h1,h2,etat)
wt2 = wT(h3,h5,etat)
wp1 = wP(h7,h6,etap)

'''
Geting relationship between mass flow rates of primary / secondary / intermediate loops

m1 = mass flow rate at left loop 
m2 = mass flow rate at right loop 
mi = mass flow rate at intermediate loop 
x2 = steam quality @ node 2

m2 = m1*x2
mi = m1*(1-x2)
'''
x2 = props.x_ph(1.380,h1-wt1) #adjusting for h2 from h2s
h8 = (1-x2)*h4 + x2*(h6-wp1) #adjusting for h7 from h7s, (mi*h4 + m2*h7) / m1 --- CV over OFWH 
s8 = props.s_ph(1.380,h8)
h9 = props.h_ps(6.89,s8)

'''
Final work / Q
'''
wp2 = wP(h9,h8,etap)
qr = h1-(h8-wp2) #adjust for h9 from h9s

'''
Output
'''
therm_efficiency = (wt1+x2*wt2-x2*wp1-wp2)/(qr) #change m2 to x2*m1, cancel out m1
print('Thermal Efficiency: {}'.format(therm_efficiency))