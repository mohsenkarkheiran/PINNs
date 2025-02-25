import tensorflow.compat.v1 as tf #'2.16.2'
#tf.enable_eager_execution()

import numpy as np
import deepxde as dde #'1.13.0'
#from functools import partial


class Heat_PINN():
    
    def __init__(self,
                 a,
                 layers = [2,30,30,1],
                 initializer = 'He normal',
                 activation = 'tanh',
                 optimizer = 'adam',
                 learning_rate = 0.001,
                 metric = ["l2 relative error"],
                 num_domain = 1000,
                 num_boundary = 0,
                 num_test = 200,
                 iters = 2000,
                 weights = []
                 ):
        self.a = a
        self.x = np.linspace(0,1,100)
        self.t = np.linspace(0,1,100)
        self.layers = layers
        self.initializer = initializer
        self.activation = activation
        self.optimizer = optimizer
        self.lr = learning_rate
        self.metric = metric
        self.num_domain = num_domain
        self.num_test = num_test
        self.num_boundary = num_boundary
        self.iterations = iters
        self.weights = None
        
    def exact_solution(self, t, x):
      
        a = self.a
        return x**2 + 2*a*t + 1
    def ref_solution(self,z):
        t = z[:,0:1]
        x = z[:,1:2]
        a = self.a
        return x**2 + 2*a*t + 1
    
    def solve(self, anchors = None, param_tune = False):
        
        x = self.x
        t = self.t
        X, T = np.meshgrid(x, t)
        a0 = self.a
        a = tf.Variable(self.a )
        geom = dde.geometry.Rectangle((0,0), (1, 1))
        x_begin = 0; x_end = 1
        def boundary_bottom(z,on_boundary):  
            return dde.utils.isclose(z[1],x_begin)
        def boundary_top(z,on_boundary):  
            return dde.utils.isclose(z[1],x_end)
        def ic_begin(z,on_boundary):
            return dde.utils.isclose(z[0],0)

       
    
        bc_bottom = dde.icbc.DirichletBC (geom, lambda z: 2*a0*z[:,0:1] + 1, boundary_bottom)
        bc_top    = dde.icbc.DirichletBC (geom, lambda z: 2*a0*z[:,0:1] + 2, boundary_top)
        bc_ic     = dde.icbc.DirichletBC (geom, lambda z: (z[:,1:2])**2 + 1, ic_begin)
    
        def HEAT_deepxde(z,w):
            dw_dt = dde.grad.jacobian(w,z,0,0)
            dw_dx = dde.grad.jacobian(w,z,0,1)
            d2w_dx2 = dde.grad.jacobian(dw_dx,z,0,1)
    
            return dw_dt - a * d2w_dx2


        if param_tune == True:
            parameters = [dde.callbacks.VariableValue(a, period=500)]
            points, ys = anchors
            observe = dde.icbc.PointSetBC(points, ys )
            bcs = [bc_top,bc_bottom,bc_ic, observe]
            anchors = points
            #weights = self.weights
            #weights.append(weights[-1])
        else: 
            parameters = []
            bcs = [bc_top,bc_bottom,bc_ic]
            anchors = None
            #weights = self.weights
            
            
        data = dde.data.PDE(geom, HEAT_deepxde,bcs,
                        #solution=self.exact_solution,
                        num_domain = self.num_domain,
                        num_boundary = self.num_boundary,
                        num_test = self.num_test,
                        anchors = anchors
                       )

        net = dde.nn.FNN(self.layers, self.activation, self.initializer)
        model = dde.Model(data, net)
        model.compile(self.optimizer, lr = self.lr, metrics = [],loss_weights = None, verbose = 0 )
        losshistory, train_state = model.train(iterations = self.iterations , display_every = 1000, verbose = 0, callbacks=parameters )
        
       
        y_pred = model.predict(np.stack((T.ravel(), X.ravel()), axis=-1)).reshape(len(t), len(x))
        
        if param_tune == False:
            return x,t,y_pred
        else:
            return [x,t,y_pred] , parameters[0].value
        
    def find_param(self, param_star, num_pts, noise_amp = 0.):
            
        pts = np.random.uniform(0, 1, [num_pts, 2])
        noise = noise_amp*np.random.uniform(-1, 1, [num_pts, 1])
        anchors =  pts, pts[:,0:1]**2 + 2*param_star*pts[:,1:2] + 1 + noise
        solution = self.solve(anchors = anchors, param_tune=True)        
        param_optim = solution[1]
            
        return param_optim
    


























