import numpy as np 
from sympy import *
import cmath
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

#------------------------------------------------------
#defining nice axis labels using latex pi and fractions
#------------------------------------------------------
def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

x, y = symbols('x y')

#----------------------------------------------
#the hamiltonian we want to compute the flow of
#----------------------------------------------
#h = ((x**4+y**4)**(1/2))*(x**2+y**2)/((x**32+y**32)**(1/16)) #a good one 
#h = (x**6+6*y**6)**(1/3)-(x**4+y**4)**(1/2) #non-globally describable example
#h = ((1/4)*x**6+24*y**6)**(1/3)-(x**2+y**2) #faster degenerating example
h = (x**26+y**26)**(1/13)

h_x = lambdify([x,y],diff(h,x)) #partial derivatives of h, use lambdify to make them easily evaluateable
h_y = lambdify([x,y],diff(h,y))


#---------------------------
#plot settings (4 subgraphs)
#---------------------------
fig = plt.figure(num = "Phase Transformations and Radius Scale Factor", figsize = (8, 8))
fig.suptitle("Canonical Transformations of $\mathbb{R}^2\setminus\{0\}$")

ax = plt.subplot2grid((2,2),(0,0)) #the angle transformation
plt.tight_layout(pad=2.0)
ax.set_xbound(0,2*np.pi)
ax.set_ybound(0,2*np.pi)
ax.set_aspect(1.0)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 16))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 16))
ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax.set_title("Transformation of the angle $\phi$")
ax.grid(True)
ax.set_xlabel('$\phi$')
ax.set_ylabel('$\widetilde{\phi}(\phi)$',rotation=0)

ax2 = plt.subplot2grid((2,2),(1,1)) #radius scale factor
plt.tight_layout(pad=2.0)
ax2.set_xbound(0,2*np.pi)
ax2.set_ybound(0,6)
ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 16))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax2.grid(True)
ax2.set_xlabel('$\phi$')
ax2.set_ylabel('$s(\phi)$',rotation=0)
ax2.set_title("The scale-factor $s(\phi)$ for the radius $r$")


ax3 = plt.subplot2grid((2,2),(0,1)) #derivative of angle transformation
plt.tight_layout(pad=2.0)
ax3.set_xbound(0,2*np.pi)
ax3.set_ybound(0,2)
ax3.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax3.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 16))
ax3.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax3.grid(True)
ax3.set_xlabel('$\phi$')
ax3.set_ylabel("$\widetilde{\phi}(\phi)'$",rotation=0)
ax3.set_title("Derivative of $\widetilde{\phi}$")


ax4 = plt.subplot2grid((2,2),(1,0)) #parametric plot of the transformed plane/circle
plt.tight_layout(pad=2.0)
ax4.set_aspect(1.0)
ax4.set_xbound(-2,2)
ax4.set_ybound(-2,2)
ax4.grid(True)
ax4.set_xlabel('$x$')
ax4.set_ylabel('${y}$',rotation=0)
ax4.set_title("Parametric Plot")


#-----------------------------------------------
#settings and structures used for the animations 
#-----------------------------------------------
ts = 1/1200 #timestep
c_x = []
c_y = []
p_x = []
p_y = []
p = 500 #discrete points on the circle
angles = []
for i in range(0,p):
        angles.append((i/p)*2*pi)
for a in angles: 
        c_x.append(cmath.exp(a*1j).real)
        c_y.append(cmath.exp(a*1j).imag)

radius = []

phaseplot, = ax.plot([],[])
radiusplot, = ax2.plot([],[])
phiprimeplot, = ax3.plot([],[])
paramplot, = ax4.plot([],[])

def init():
    phaseplot.set_data([],[])
    radiusplot.set_data([],[])
    phiprimeplot.set_data([],[])
    paramplot.set_data([],[])
    return phaseplot, radiusplot, phiprimeplot, paramplot,

def animate(self):
    global c_x,c_y,p_x,p_y,h_x,h_y,angles,ts #make accessed variables global

    p_x = [] 
    p_y = []
    new_angles = []
    new_radius = []
    phi_prime = []
    for j in range(len(angles)): 
        p_x.append(c_x[j] + ts*h_y(c_x[j],c_y[j])) #move the points according to the flow of the hamiltonian vector field of h
        p_y.append(c_y[j] - ts*h_x(c_x[j],c_y[j]))
        new_angles.append(np.arctan2(p_y[j],p_x[j])+ -1*(np.sign(np.arctan2(p_y[j],p_x[j]))-1)*np.pi) #record angles of the points after moving them 
        norm = (p_y[j]**2+p_x[j]**2)**(1/2) #record norm of the points after moving them
        new_radius.append(norm)
        phi_prime.append((1/norm)**(1/2)) #cheat: use math to avoid numerical calculation of the derivative of \widetilde{\phi} 
                                          
    c_x=p_x
    c_y=p_y
    
    phaseplot.set_data(angles,new_angles) #set the new data to the plots
    radiusplot.set_data(angles,new_radius)
    phiprimeplot.set_data(angles,phi_prime)
    paramplot.set_data(c_x,c_y)
    
    return phaseplot, radiusplot, phiprimeplot, paramplot,

phase_anim = animation.FuncAnimation(fig,animate,init_func=init,frames=20000,blit=false,interval=1) #'live' animation of the flow

figManager = plt.get_current_fig_manager() #open in full-screen
figManager.full_screen_toggle()

plt.show()
