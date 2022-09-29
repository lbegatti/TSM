from array import array
import numpy as np

class NelsonSiegel:
    '''
    Class for Nelson Siegel Calculated in question 4
    '''

    def __init__(self, Lambda:float, sigmaL:float, sigmaS:float, Xt:np.array) -> None:
        self.Xt=Xt
        self.Lambda=Lambda #Lambda with cap since lambda is reserved
        self.sigmaL=sigmaL
        self.sigmaS=sigmaS
    
    def A(self, tau)->float:
        #return np.square(self.sigmaL)*np.power(tau,3)/6+np.square(self.sigmaS)/(4*np.power(self.Lambda,3))*(2*tau*self.Lambda-3+4*np.exp(-tau*self.Lambda)-np.exp(-2*tau*self.Lambda))
        bracket=tau/(2*np.square(self.Lambda))-(1-np.exp(-self.Lambda*tau))/np.power(self.Lambda,3)+(1-np.exp(-2*self.Lambda*tau))/(4*np.power(self.Lambda,3))
        return np.square(self.sigmaL)*np.power(tau,3)/6+np.square(self.sigmaS)*bracket

    def B(self, tau)->np.array:
        return np.array([-tau,(np.exp(-self.Lambda*tau)-1)/self.Lambda])
    
    def getYields(self, tau:list):
        return [(t, -self.A(t)/t-np.dot(self.B(t),self.Xt)/t) for t in tau]




