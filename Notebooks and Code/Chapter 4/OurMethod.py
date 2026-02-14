import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import torch
import mctorch.nn as mnn
import mctorch.optim as moptim




# if we can use regression, try this:
def regress(B, y, criterion=torch.nn.MSELoss(), optimizer=moptim.rSGD, lr=1e-2, n_epochs=5000):
    manifold_param = mnn.Parameter(manifold=mnn.Euclidean(B.shape[1]))
    Btensor = torch.tensor(B).float()
    
    def model(mat):
        term1 = torch.matmul(Btensor, mat.float())
        return term1
    
    y = torch.tensor(y).float()
    
    optimizer = optimizer(params = [manifold_param], lr=lr)
    for epoch in range(n_epochs):
        y_pred = model(manifold_param)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return loss.item(), manifold_param
    
    
    
# Here we can use level set estimation if regression is not appropriate.
def tryDim(B, d, n_epochs=5000, criterion=torch.nn.MSELoss(), optimizer=moptim.rSGD, lr=1e-2):
    manifold_param = mnn.Parameter(manifold=mnn.Stiefel(B.shape[1],d))
    Btensor = torch.tensor(B).float()
    
    def model(mat):
        term1 = torch.matmul(Btensor, mat)
        return term1
    
    y = torch.zeros((B.shape[0],d))
    
    optimizer = optimizer(params = [manifold_param], lr=lr)
    for epoch in range(n_epochs):
        y_pred = model(manifold_param)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return loss.item(), manifold_param



class invF():
    def __init__(self,W,feats,Vars,X):
        self.ds = X
        self.params = W
        self.feats = feats
        self.Vars= Vars
        #self.B = self.featFunc.fit_transform(self.ds)
        
        
    def constructF(self):
        from sympy import symbols
        from sympy import lambdify
        #from sympy import Matrix
        # Create sympy symbols
        exec(",".join(self.Vars) + '= symbols(" ".join(self.Vars))')
        p1 = ",".join(self.feats)
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = list(eval(p3))
        F = lambdify([self.Vars],p3a)
        return F


    def getFeatMat(self, X):
        F = self.constructF()
        B = np.array([F(p) for p in X])
        B = torch.tensor(B).float()
        return B
    
    
    
    def f(self,Y):
        w = self.params
        trans =  self.getFeatMat(self.ds) #self.featFunc.transform(Y)
        term = torch.matmul(trans, w) #trans@w
        return term
    
    
    def constructJ(self):
        from sympy import symbols
        from sympy import lambdify
        from sympy import Matrix
        # Create sympy symbols
        exec(",".join(self.Vars) + '= symbols(" ".join(self.Vars))')
        p1 = ",".join(self.feats)
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = list(eval(p3))
        p4 = Matrix(p3a)
        J = p4.jacobian(Matrix(self.Vars))
        Jnp = lambdify([self.Vars],J)
        return Jnp    
    
    
    def Jacf(self,X):
        from sympy import symbols
        from sympy import lambdify
        from sympy import Matrix
        # Create sympy symbols
        exec(",".join(self.Vars) + '= symbols(" ".join(self.Vars))')
        p1 = ",".join(self.feats)
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = list(eval(p3))
        p4 = Matrix(p3a)
        J = p4.jacobian(Matrix(self.Vars))
        Jnp = lambdify([self.Vars],J)
        
        jacs = []
        for _x in X:
            jacs.extend([torch.matmul(torch.transpose(torch.tensor(Jnp(_x)).float(), 0, 1), self.params)])
            
        if len(jacs[0].shape)==1:
            jacs = [torch.transpose(jac.reshape((-1,1)),0,1) for jac in jacs]
        else:
            jacs = [torch.transpose(jac,0,1) for jac in jacs]
        bigJ = torch.stack(jacs, dim=0)
        return bigJ    




    
    
# This conveniently allows us to define a function and (symbolically) compute the Jacobian.    
class invf():
    def __init__(self,W,featFunc,X):
        self.ds = X
        self.params = W
        self.featFunc = featFunc
        self.B = self.featFunc.fit_transform(self.ds)
        
        
    def f(self,Y):
        w = self.params
        trans = self.featFunc.transform(Y)
        term = trans@w
        return term
    
    
    def jacf(self,x): #Jacobian at a single point.
        #w = self.params
        dim = self.featFunc.n_features_in_
        from sympy import symbols
        from sympy import Matrix
        from sympy import lambdify
        # create sympy symbols.
        exec(",".join(list(self.featFunc.get_feature_names_out()[1:dim+1])) + '= symbols(" ".join(list(self.featFunc.get_feature_names_out()[1:dim+1])))')
        # Fix things: no x0^2 and such.
        p1 = ",".join(list(self.featFunc.get_feature_names_out()))
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = list(eval(p3))
        p4 = Matrix(p3a)
        # Define the jacobian and make it a numpy array.
        J = p4.jacobian(Matrix(p3a[1:dim+1]))
        #print(J)
        Jnp = lambdify([p3a[1:dim+1]],J)
        #print(self.params.transpose()@J)
        realJ = self.params.transpose()@Jnp(x)
        return realJ


    def JacFunc(self,X): #stacked Jacobian matrix at each point in X
        #w = self.params
        dim = self.ds.shape[1] #self.featFunc.n_features_in_
        from sympy import symbols
        from sympy import Matrix
        from sympy import lambdify
        # create sympy symbols.
        #exec(",".join(list(self.featFunc.get_feature_names_out()[1:dim+1])) + '= symbols(" ".join(list(self.featFunc.get_feature_names_out()[1:dim+1])))')
        exec(",".join(self.featFunc.get_feature_names_out()[1:dim+1]) + '= symbols(" ".join(self.featFunc.get_feature_names_out()[1:dim+1]))')
        # Fix things: no x0^2 and such.
        p1 = ",".join(self.featFunc.get_feature_names_out())
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = eval(p3)
        p4 = Matrix(p3a)
        # Define the jacobian and make it a numpy array.
        J = p4.jacobian(Matrix(p3a[1:dim+1]))
        Jnp = lambdify([p3a[1:dim+1]],J)
        return Jnp
        
        
    
    def Jacf(self,X): #stacked Jacobian matrix at each point in X
        #w = self.params
        dim = self.ds.shape[1] #self.featFunc.n_features_in_
        from sympy import symbols
        from sympy import Matrix
        from sympy import lambdify
        # create sympy symbols.
        #exec(",".join(list(self.featFunc.get_feature_names_out()[1:dim+1])) + '= symbols(" ".join(list(self.featFunc.get_feature_names_out()[1:dim+1])))')
        exec(",".join(self.featFunc.get_feature_names_out()[1:dim+1]) + '= symbols(" ".join(self.featFunc.get_feature_names_out()[1:dim+1]))')
        # Fix things: no x0^2 and such.
        p1 = ",".join(self.featFunc.get_feature_names_out())
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = eval(p3)
        p4 = Matrix(p3a)
        # Define the jacobian and make it a numpy array.
        J = p4.jacobian(Matrix(p3a[1:dim+1]))
        Jnp = lambdify([p3a[1:dim+1]],J)        
        
        
        jacs = []
        for x in X:
            jacs.extend([torch.matmul(torch.transpose(torch.tensor(Jnp(x)).float(), 0, 1), self.params)])
            
        if len(jacs[0].shape)==1:
            jacs = [torch.transpose(jac.reshape((-1,1)),0,1) for jac in jacs]
        else:
            jacs = [torch.transpose(jac,0,1) for jac in jacs]
        bigJ = torch.stack(jacs, dim=0)
        return bigJ
    
    
    def Jac2f(self,X, ex=1): #stacked Jacobian matrix at each point in X
        #w = self.params
        dim = self.ds.shape[1] #self.featFunc.n_features_in_
        from sympy import symbols
        from sympy import Matrix
        from sympy import lambdify
        # create sympy symbols.
        #exec(",".join(list(self.featFunc.get_feature_names_out()[1:dim+1])) + '= symbols(" ".join(list(self.featFunc.get_feature_names_out()[1:dim+1])))')
        exec(",".join(self.featFunc.get_feature_names_out()[ex:dim+ex]) + '= symbols(" ".join(self.featFunc.get_feature_names_out()[ex:dim+ex]))')
        # Fix things: no x0^2 and such.
        p1 = ",".join(self.featFunc.get_feature_names_out())
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = eval(p3)
        p4 = Matrix(p3a)
        # Define the jacobian and make it a numpy array.
        J = p4.jacobian(Matrix(p3a[ex:dim+ex]))
        Jnp = lambdify([p3a[ex:dim+ex]],J)        
        
        
        jacs = []
        for x in X:
            #jacs.extend([self.params.transpose()@Jnp(x)])
            #jacs.extend([torch.matmul(torch.transpose(self.params), Jnp(x))])
            #print(torch.tensor(Jnp(x)))
            jacs.extend([torch.matmul(torch.transpose(torch.tensor(Jnp(x)).float(), 0, 1), self.params)])
            
        jacs = [torch.transpose(jac,0,1) for jac in jacs]
        #print(jacs[0].shape)
        bigJ = torch.stack(jacs, dim=0)
        #bigJ = np.array(jacs) #np.vstack(jacs)
        return bigJ
    
    
    def JacfSym(self,X): #stacked Jacobian matrix at each point in X
        #w = self.params
        dim = self.ds.shape[1] #self.featFunc.n_features_in_
        from sympy import symbols
        from sympy import Matrix
        from sympy import lambdify
        # create sympy symbols.
        #exec(",".join(list(self.featFunc.get_feature_names_out()[1:dim+1])) + '= symbols(" ".join(list(self.featFunc.get_feature_names_out()[1:dim+1])))')
        exec(",".join(self.featFunc.get_feature_names_out()[1:dim+1]) + '= symbols(" ".join(self.featFunc.get_feature_names_out()[1:dim+1]))')
        # Fix things: no x0^2 and such.
        p1 = ",".join(self.featFunc.get_feature_names_out())
        p2 = p1.replace(" ", "*")
        p3 = p2.replace("^", "**")
        p3a = eval(p3)
        p4 = Matrix(p3a)
        # Define the jacobian and make it a numpy array.
        J = p4.jacobian(Matrix(p3a[1:dim+1]))
        return J
    
    
    def getfunc(self):
        return self.f
    
    
    def getgrad(self):
        return self.gradf
    
    
    def getParams(self):
        return self.params
    
    
    def getDS(self):
        return self.ds
        
        
        
        
# Once the Jacobian is in hand, we can apply the "convolution" needed before looking for infinitesimal generators.
# This is a memory hog. Can we fix this later?
def getExtendedFeatureMatrix2(B,J,dim):
    n = B.shape[0]
    m = B.shape[1]
    ExtArray = []
    for j in range(n):
        newBthing = np.append(B[j],np.zeros((dim-1)*m))
        Barray = np.array([np.roll(newBthing,i*m) for i in range(dim)])
        Barray = torch.tensor(Barray).float()
        thing = torch.matmul(J[j], Barray)
        ExtArray.extend(thing)
        
    ExtArray = torch.stack(ExtArray, dim=0)
    return ExtArray



# Here is optimizing for the generators.

def tryDimV(extB, d, n_epochs=5000, criterion=torch.nn.MSELoss(), optimizer=moptim.rSGD, lr=1e-2):
    manifold_param = mnn.Parameter(manifold=mnn.Stiefel(extB.shape[1],d))
    Btensor = extB.detach() #torch.tensor(B).float()
    
    def model(mat):
        term1 = torch.matmul(Btensor, mat)
        return term1
    
    y = torch.zeros((extB.shape[0],d))
    
    optimizer = optimizer(params = [manifold_param], lr=lr)
    for epoch in range(n_epochs):
        y_pred = model(manifold_param)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return loss.item(), manifold_param


