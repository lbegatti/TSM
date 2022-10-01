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
        bracket=2*tau*self.Lambda-3+4*np.exp(-tau*self.Lambda)-np.exp(-2*tau*self.Lambda)
        return (np.square(self.sigmaL)*np.power(tau,3))/6+np.square(self.sigmaS)/(4*np.power(self.Lambda,3))*bracket

    def B(self, tau)->np.array:
        return np.array([-tau,(np.exp(-self.Lambda*tau)-1)/self.Lambda])
    
    def getYields(self, tau:list)->list:
        return [(t, -self.A(t)/t-np.dot(self.B(t),self.Xt)/t) for t in tau]




