import math
import random
import numpy as np
from micrograd import Value

class Module:
     def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

     def parameters(self):
        return []
     
class Neuron(Module):

    def __init__(self,nin):  #nonlin=True?                                #NOTE:nin = number of inputs neurons
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        # self.nonlin = nonlin
    
    def __call__(self,x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w,x)), self.b)             #NOTE: if comes up a lot make this _.dot(_)
        # out = act.relu()                                                 #NOTE: for examples so far this doesnt work well                                              
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b] #concatination 
    
    def __repr__(self):
        return f"{'ReLU' }Neuron({len(self.w)})"                           

class Layer(Module):
    
    def __init__(self, nin, nout): #**kwargs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neurons in self.neurons for p in neurons.parameters()]  

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]   #nonlin=i!=len(nouts)-1)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
