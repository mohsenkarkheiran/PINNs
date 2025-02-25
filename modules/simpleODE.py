import tensorflow.compat.v1 as tf #'2.16.2'
#tf.enable_eager_execution()

import numpy as np
import deepxde as dde #'1.13.0'


class ODE_PINN():
    
    def __init__(self,
                 omega,
                 layers = [1,30,30,1],
                 initializer = 'He normal',
                 activation = 'tanh',
                 optimizer = 'adam',
                 learning_rate = 0.001,
                 metric = ["l2 relative error"],
                 num_domain = 1000,
                 num_boundary = 0,
                 num_test = 200,
                 iters = 2000
                 ):
        self.omega = omega
        self.x = np.linspace(-np.pi, np.pi, 1000)
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
    def exact_solution(self):
        
        omega = self.omega
        return (1/omega)*np.sin(omega * self.x)
    
    def solve(self, anchors = None, param_tune = False):
        x = self.x
        omega = tf.Variable(self.omega )
        geom = dde.geometry.TimeDomain(x[0], x[-1])

        x_begin = 0; y_begin = 0
        def boundary_begin(x,_):  # Since the equation is of order one, we only need one boundary/initial condition.
                          # In priciple one coulde apply the initial condion to the solutions by "Hard constrains",
                          # But for this example we decided to use it explicitly as follows:
            return dde.utils.isclose(x[0],x_begin)

        def bc_func_begin(x,y,_): 
    
            return y - y_begin

        
        def ODE_deepxde(x,y):
            dy_dx = dde.grad.jacobian(y,x)
    
            return dy_dx - tf.cos(omega*x)
        
        bc1 = dde.icbc.OperatorBC(geom,bc_func_begin,boundary_begin) 
        
        if param_tune == True:
            parameters = [dde.callbacks.VariableValue(omega, period=500)]
            points, ys = anchors
            observe = dde.icbc.PointSetBC(points, ys )
            bcs = [bc1, observe]
            anchors = points
            
        else: 
            parameters = []
            bcs = [bc1]
            anchors = None
           
        
        data = dde.data.PDE(geom, ODE_deepxde,bcs, 
                        num_domain = self.num_domain,
                        num_boundary = self.num_boundary, # Note that no boundary points is choosen. This is harmless since the ODE
                                          # is of first order, and we already applied the initial conditions
                        num_test = self.num_test,
                        anchors = anchors)
        
        
       
        net = dde.nn.FNN(self.layers, self.activation, self.initializer)
        model = dde.Model(data, net)
        model.compile(self.optimizer, lr = self.lr, verbose = 0 )
        losshistory, train_state = model.train(iterations = self.iterations , display_every = 1000, verbose = 0, callbacks=parameters )
        
        y_pred = model.predict(x[:, None])
        
        if param_tune == False:
            return x,y_pred
        else:
            return [x,y_pred] , parameters[0].value
            
    def find_param(self, param_star, num_pts, noise_amp = 0.):
        
        x = self.x
        pts = np.random.uniform(x[0], x[1], [num_pts, 1])
        noise = noise_amp*np.random.uniform(-1, 1, [num_pts, 1])
        anchors =  pts, np.sin(param_star*pts)/param_star + noise

        solution = self.solve(anchors = anchors, param_tune=True)        
        param_optim = solution[1]
        
        return param_optim
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    