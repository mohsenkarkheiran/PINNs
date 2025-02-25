import tensorflow.compat.v1 as tf #'2.16.2'
#tf.enable_eager_execution()

import numpy as np
import deepxde as dde #'1.13.0'
#from functools import partial

class PDE_PINN():
    
    def __init__(self,
                 alpha = -1.,
                 beta = 0,
                 gamma = 1,
                 delta = -2,
                 layers = [2,30,30,2],
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
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
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
        
    def v_sol(self, X, T):
        return (X**2 + T + 1)*np.exp(-T)
    def w_sol(self, X, T):
        return (X**2 + T + 1)*(np.exp(-T) - np.exp(-2*T))
    
    def solve(self, anchors = None, param_tune = False):
        
        x = self.x
        t = self.t
        X, T = np.meshgrid(x, t)
        geom = dde.geometry.Rectangle((0,0), (1, 1))
        x_begin = 0; x_end = 1
        def boundary_bottom(z,on_boundary):  
            return dde.utils.isclose(z[1],x_begin)
        def boundary_top(z,on_boundary):  
            return dde.utils.isclose(z[1],x_end)
        def ic_begin(z,on_boundary):
            return dde.utils.isclose(z[0],0)


        a = 0.5

        bc_bottom_v = dde.icbc.DirichletBC (geom, lambda z: [(2*a*z[:,0:1] + 1)*tf.exp(-z[:,0:1])], boundary_bottom, component = 0)
        bc_bottom_w = dde.icbc.DirichletBC (geom, lambda z: [(2*a*z[:,0:1] + 1)*(tf.exp(-z[:,0:1])-tf.exp(-2*z[:,0:1]))],boundary_bottom, component = 1)
        
        bc_top_v    = dde.icbc.DirichletBC (geom, lambda z: [(2*a*z[:,0:1] + 2)*tf.exp(-z[:,0:1])],boundary_top, component = 0)
        bc_top_w    = dde.icbc.DirichletBC (geom, lambda z: [(2*a*z[:,0:1] + 2)*(tf.exp(-z[:,0:1])-tf.exp(-2*z[:,0:1]))],boundary_top, component = 1)

        bcs = [bc_top_v, bc_top_w, bc_bottom_v, bc_bottom_w]
            
            
        alpha  = tf.Variable(self.alpha )
        beta   = tf.Variable(self.beta )
        gamma  = tf.Variable(self.gamma )
        delta  = tf.Variable(self.delta )
            
        def PDE_deepxde(z,y):
    
            v = y[:,0:1]
            w = y[:,1:2]
    
            dw_dt = dde.grad.jacobian(w,z,0,0)
            dw_dx = dde.grad.jacobian(w,z,0,1)
            d2w_dx2 = dde.grad.jacobian(dw_dx,z,0,1)
            
            dv_dt = dde.grad.jacobian(v,z,0,0)
            dv_dx = dde.grad.jacobian(v,z,0,1)
            d2v_dx2 = dde.grad.jacobian(dv_dx,z,0,1)
        
            return [
                dv_dt - a * d2v_dx2 - alpha*v - beta*w,
                dw_dt - a * d2w_dx2 - gamma*v - delta*w
                ]
        
        if param_tune == True:
            parameters = [dde.callbacks.VariableValue([alpha,beta,gamma,delta], period = 200)]
            points, Zs = anchors
            obs_v = dde.icbc.PointSetBC(points, Zs[:,0:1],component = 0)
            obs_w = dde.icbc.PointSetBC(points, Zs[:,1:2],component = 1)
            bcs.extend([obs_v,obs_w])
            anchors = points
                
        else: 
            parameters = []
            bcs = bcs
            anchors = None
                
        def output_transform(z, q):
            v = q[:, 0:1]
            w = q[:, 1:2]
            t = z[:, 0:1]
            x = z[:, 1:2]
            return tf.concat([v * tf.tanh(t) + x**2+1, w * tf.tanh(t)], axis=1)
        
        data = dde.data.PDE(geom, PDE_deepxde,bcs,                           
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
            
        Z_pred = model.predict(np.stack((T.ravel(), X.ravel()), axis=-1)).reshape(len(t), len(x),2) 
        
        if param_tune == False:
            return x,t,Z_pred
        else:
            return [x,t,Z_pred] , parameters[0].value
        
        
    def find_param(self, num_pts, noise_amp = 0.):
        
        pts = np.random.uniform(0, 1, [num_pts, 2])
        noise_v = noise_amp*np.random.uniform(-1, 1, [num_pts, 1])
        noise_w = noise_amp*np.random.uniform(-1, 1, [num_pts, 1])
        v = self.v_sol(pts[:,:1], pts[:,1:]) + noise_v
        w = self.w_sol(pts[:,:1], pts[:,1:]) + noise_w
            
        anchors =  pts, np.stack([v.flatten(), w.flatten()], 1)
        solution = self.solve(anchors = anchors, param_tune=True)        
        param_optim = solution[1]
                
        return param_optim
        
        
        
        
        
        
        
        
        