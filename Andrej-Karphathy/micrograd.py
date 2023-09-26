import math
import numpy as np


class Value:
    ''' Stores a single scalar value and its gradient '''
    
    def __init__(self, data, children=(), _op='', label=''): 
        self.data = data
        self.grad = 0.0
        self._prev=set(children)
        self._op = _op
        self._backward = lambda: None
        self.label= label
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):   
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += out.grad                                                                      #+= instead of = for cases like  a+a, or any time a varaible is used more then once
            other.grad += out.grad
        out._backward = _backward

        return out 
    
    def __mul__(self, other):   
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():    
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward

        return out 

    def __pow__(self, other): #Value * number
        # assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        if isinstance(other,Value):
            out = Value(self.data**other.data, (self,other), '*')

            def _backward():
                self.grad  += (other.data * self.data**(other.data-1)) * other.grad * out.grad             #NOTE: dont think the other.grad will be defined yet. so would we have to add an implict solver?
                other.grad += ((self.data ** other.data) * np.log(self.data))* self.grad  * out.grad
            out._backward = _backward

        else:
            out = Value(self.data**other, (self, ), f'**{other}')              

            def _backward():    
                self.grad  += (other * self.data**(other-1)) * out.grad               
            out._backward = _backward

        return out 

    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward

        return out
    ##what else to add power of both values, -,+,* with only one value. then the sin exp ..

    def sin(self):
        x = self.data
        out = Value(math.sin(x), (self,), 'sin')

        def _backward():
            self.grad += math.cos(x) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        x = self.data
        out = Value(math.cos(x), (self,), 'cos')

        def _backward():
            self.grad += math.sin(x) * out.grad
        out._backward = _backward
        return out
    
    def tan(self):
        x = self.data
        out = Value(math.tan(x), (self,),'tan')

        def _backward():
            self.grad += (math.sec(x)**2) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,),'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value( 0 if self.data <0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += ( out.data > 0) * out.grad                                                     # NOTE: out.data>0 is 1 if true and 0 f false, hence gives what we want
        out._backward = _backward

        return out
    
    def log(self):
        x = self.data
        out = Value(np.log(x), (self,), 'log')

        def _backward():
            self.grad += (1/x) * out.grad
        out.grad = _backward

        return out

    #operations that are sepical cases of previous operations 

    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __neg__(self): # -self
        return self * -1

   

    # if the order is the other way around, e.i 2 * a

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return other * self**-1


    def backward(self):
        #topological sort 'right to left'
        topo = []
        visited = set()
     
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
                
        build_topo(self)

        #the first variable will always have grad 1
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
