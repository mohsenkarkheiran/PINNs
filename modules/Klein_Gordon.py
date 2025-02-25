import tensorflow.compat.v1 as tf #'2.16.2'
#tf.enable_eager_execution()

import numpy as np
import deepxde as dde #'1.13.0'
#from functools import partial


class KG_PINN():
    
    def __init__(self,
                 a,
                 b,
                 case,
                 t_max,
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
        self.t_max = t_max
        self.a = a
        self.b = b
        self.case = case
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
        
    def exact_sol(self,x,t):
        
        b = self.b
        a = self.a
        
        if self.case == 'Case1': 
            mu = np.sqrt(b + a**2 * np.pi**2 / 4)
            return np.cos(x*np.pi/2)*(a*np.cos(mu*t)+b*np.sin(mu*t))
        elif self.case == 'Case2':
            mu = np.sqrt(b - a**2 * np.pi**2 / 4)
            return np.exp(-(np.pi/2)*x)*(a*np.cos(mu*t)+b*np.sin(mu*t))
    
    def solve(self, anchors = None, param_tune = False):
        
        case = self.case
        a_b = tf.constant(self.a) #paramters on the boundary should be fixed
        b_b = tf.constant(self.b)
        a = tf.Variable(self.a) #parameters on the operators
        b = tf.Variable(self.b)
        if case == "Case1":
            x_min = 1; x_max = 5
            mu = tf.math.sqrt(b_b + a_b**2 * np.pi**2 / 4)
            
        elif case == "Case2":
            x_min = 0; x_max = 4
            mu = tf.math.sqrt(b_b - a_b**2 * np.pi**2 / 4)
            
        t_min = 0;
        t_max = self.t_max;
        
        x = np.linspace(x_min, x_max, 100)
        t = np.linspace(t_min, t_max, 100)
        X, T = np.meshgrid(x, t)


        geom = dde.geometry.Rectangle((x_min,t_min), (x_max,t_max))
        
        def boundary_bottom(z,on_boundary):  
            return dde.utils.isclose(z[0],x_min)

        def boundary_top(z,on_boundary):  
            return dde.utils.isclose(z[0],x_max)

        def ic_begin(z,on_boundary):
            return dde.utils.isclose(z[1],0)

        if case == 'Case1':
            bc_min_x = dde.icbc.DirichletBC (geom, lambda z: 0, boundary_bottom, component = 0)
            bc_max_x = dde.icbc.DirichletBC (geom, lambda z: 0, boundary_top,    component = 0)
            bc_t_min_N = dde.icbc.NeumannBC(geom,lambda z: -b_b*mu*tf.cos(((np.pi)/2)*z[:,0:1]),ic_begin,component=0)
        
        elif case == 'Case2':
            bc_min_x = dde.icbc.DirichletBC (geom, lambda z: a_b*tf.cos(mu*z[:,1:2])+b_b*tf.sin(mu*z[:,1:2]), boundary_bottom, component = 0)
            bc_max_x = dde.icbc.DirichletBC (geom, lambda z: tf.exp(-2*np.pi)*(a_b*tf.cos(mu*z[:,1:2])+b_b*tf.sin(mu*z[:,1:2])), boundary_top,    component = 0)
            bc_t_min_N = dde.icbc.NeumannBC(geom,lambda z: -b_b*mu*tf.exp(-(np.pi/2)*z[:,0:1]),ic_begin,component=0)

        bcs = [bc_min_x, bc_max_x, bc_t_min_N]
        
        def KG_deepxde(z,y):
    
            psi = y[:, 0:1]
    
            dpsi_dt = dde.grad.jacobian(psi,z,0,1)
            d2psi_dt2 = dde.grad.jacobian(dpsi_dt,z,0,1)
            dpsi_dx = dde.grad.jacobian(psi,z,0,0)
            d2psi_dx2 = dde.grad.jacobian(dpsi_dx,z,0,0)
    
    
            return d2psi_dt2  - (a**2)* d2psi_dx2 + b*psi
        
        def output_transform(z, q): # Here we applied the initial conditions as 'Hard constraints'
            
            case = self.case
            psi = q[:, 0:1]
    
            x = z[:, 0:1]
            t = z[:, 1:2]
            if case == 'Case1':
                return psi * tf.tanh(t) + a_b*tf.cos(((np.pi)/2)*x) 
            if case == 'Case2':
                return psi * tf.tanh(t) + a_b*tf.exp(-(np.pi/2)*x)
        
        if param_tune == True:
            parameters = [dde.callbacks.VariableValue([a,b], period=500)]
            points, ys = anchors
            observe = dde.icbc.PointSetBC(points, ys )
            bcs = [bc_min_x, bc_max_x, observe]
            anchors = points
            
        else: 
            parameters = []
            bcs = bcs
            anchors = None
        
        data = dde.data.PDE(geom, KG_deepxde,bcs,
                            num_domain = self.num_domain,
                            num_boundary = self.num_boundary,
                            num_test = self.num_test,
                            anchors = anchors
                           )

        net = dde.nn.FNN(self.layers, self.activation, self.initializer)
        net.apply_output_transform(output_transform)
            
        model = dde.Model(data, net)
        model.compile(self.optimizer, lr = self.lr, verbose = 0 )
        losshistory, train_state = model.train(iterations = self.iterations , display_every = 1000, verbose = 0, callbacks=parameters )
            
           
        y_pred = model.predict(np.stack((X.ravel(), T.ravel()), axis=-1)).reshape(len(x), len(t), 1)
        if param_tune == False:
            return x,t,y_pred
        else:
            return [x,t,y_pred] , parameters[0].value
        
    def find_param(self, num_pts, noise_amp = 0.):
        
        t_min = 0; x_min = 0
        t_max = self.t_max; x_max = 5
        
        X = np.random.uniform(x_min, x_max, num_pts)
        T = np.random.uniform(t_min, t_max, num_pts)
        
        pts = np.stack([X,T], 1)
        noise = noise_amp*np.random.uniform(-1, 1, [num_pts, 1])
        
        anchors =  pts, self.exact_sol(X, T) + noise
        solution = self.solve(anchors = anchors, param_tune=True)        
        param_optim = solution[1]
             
        return param_optim
        
        
        