#import tensorflow.compat.v1 as tf #'2.16.2'
#tf.enable_eager_execution()

import streamlit as st
import numpy as np
#import deepxde as dde #'1.13.0'
import matplotlib.pyplot as plt #'3.9.2'
#from matplotlib import colormaps

from modules import simpleODE, Heat_EQ, LinearSystem, Hamilton_Jacobi, Klein_Gordon


st.header("DeepXDE based PINN solutions")

prob = st.selectbox("Choose the model",
             ('Simple ODE','Heat Equation','Linear System of PDEs','Hamilton Jacobi','Klein Gordon'))


if prob == "Simple ODE":
    st.image("ProblemSets/SimpleODE.png", caption = "Problem explanation, ref: master notebook")
    
    
    omega = st.number_input(label = 'Give me the (initial) value of Omega',
                            min_value = 0.1,
                            max_value = 2000.,
                            value = 1.)
   
    initializer = st.selectbox("Choose the kernel_initializer",
                               ('He normal', 'He uniform', 'Glorot normal', 'Glorot uniform'))
    activation = st.selectbox('Choose the activation function',
                              ('tanh', 'elu', 'relu', 'sigmoid', 'sin'))
    optimizer = st.selectbox('Choose the optimizer',
                            ('adam', "sgd", "sgdnesterov", "adagrad", ))
   
    layers_t = st.text_input("Enter the Neural Network shape (a sequence separated by commas):",
                             "1,30,30,1")
    layers_s = layers_t.split(",")
    layers = [int(l) for l in layers_s]
    
    lr = st.slider(label = "Learning rate (in unit of 0.0001)",
                   min_value = 1.,
                   max_value = 1000.,
                   value = 10.)
    
    num_domain = st.number_input('Number of the points in the domain',
                                 min_value = 10,
                                 max_value= 10000,
                                 value = 1000)
    num_boundary = st.number_input('Number of the boundary points',
                                   min_value = 0,
                                   max_value = 5,
                                   value = 1
                                   )
    num_test = st.number_input('How many test points?',
                               min_value=1,
                               max_value=1000,
                               value =100
                               )
    iters = st.number_input('How many iterations?',
                            min_value=100,
                            max_value=10000,
                            value = 1000)
    

    pinn = simpleODE.ODE_PINN(
        omega = omega,
        layers = layers,
        initializer = initializer,
        activation = activation,
        optimizer = optimizer,
        learning_rate = lr * 0.0001,
        metric = [],
        num_domain = num_domain,
        num_boundary = num_boundary,
        num_test = num_test,
        iters = iters)
    
    sol = pinn.exact_solution()
    
    choice = st.radio(" ",
                ["PINN Solution","Paramter Identification"],
                captions = [],
                )
        
    if choice == "PINN Solution":
        
    
        if st.button('PINN Solve'):
            x, y =  pinn.solve()
            fig, ax = plt.subplots(figsize = (8,8))
            
            ax.plot(x, y, color='r',label='PINNS Prediction', ls=':')
            ax.plot(x, sol, lw=1, color='b', label='Exact Solution')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Exact and PINNS solution for omega = {}'.format(omega))            
                
            ax.legend()

            st.pyplot(fig)
        
    
    if choice == "Paramter Identification":
     
        st.text_area(' ', 
                     "In this part we will simulate an environment with background values for parameters. Suppose an experimenter collected some date points, and wishes to find the background paramters values.")
       
        param_star = st.number_input(label="What is the background parameter?",
                                     min_value = 0.1,
                                     max_value = 2000.,
                                     value = 1. )
        num_pts = st.number_input(label = "How many data points are collected?",
                                  min_value = 1,
                                  max_value = 2000,
                                  value = 100
                                  )
        noise_amp = st.number_input(label = 'If you want to creat a noisy environment, sepcify its amplitude',
                                    min_value = 0.,
                                    max_value = 1.,
                                    value = 0.) 
        
        
        if st.button('PINN Param Identification'):   
            omega_optim =  pinn.find_param(param_star, num_pts, noise_amp)[0]
        
            st.text_area('The identified value for background paramter:',
                         f'omega = {np.round(omega_optim, 4)}')
    
elif prob == 'Heat Equation':
    st.image("ProblemSets/HeatEQ.png", caption = "Problem explanation, ref: master notebook")
    
    
    a = st.number_input(label = 'Give me the (initial) value of a',
                            min_value = 0.1,
                            max_value = 2000.,
                            value = 1.)
   
    initializer = st.selectbox("Choose the kernel_initializer",
                               ('He normal', 'He uniform', 'Glorot normal', 'Glorot uniform'))
    activation = st.selectbox('Choose the activation function',
                              ('tanh', 'elu', 'relu', 'sigmoid', 'sin'))
    optimizer = st.selectbox('Choose the optimizer',
                            ('adam', "sgd", "sgdnesterov", "adagrad", ))
   
    #weights_t = st.text_input("Enter the weights of interior points and boundary/initial points (separated by commas)) :",
    #                         "1,2,2,2")
    #weights_s = weights_t.split(",")
    #weights = [int(l) for l in weights_s]
    
    layers_t = st.text_input("Enter the Neural Network shape (a sequence separated by commas):",
                             "2,30,30,1")
    layers_s = layers_t.split(",")
    layers = [int(l) for l in layers_s]
    
    lr = st.slider(label = "Learning rate (in unit of 0.0001)",
                   min_value = 1.,
                   max_value = 1000.,
                   value = 10.)
    
    num_domain = st.number_input('Number of the points in the domain',
                                 min_value = 10,
                                 max_value= 10000,
                                 value = 1000)
    num_boundary = st.number_input('Number of the boundary points',
                                   min_value = 0,
                                   max_value = 1000,
                                   value = 1
                                   )
    num_test = st.number_input('How many test points?',
                               min_value=1,
                               max_value=1000,
                               value =100
                               )
    iters = st.number_input('How many iterations?',
                            min_value=100,
                            max_value=10000,
                            value = 1000)
    

    
    pinn = Heat_EQ.Heat_PINN(
        a = a,
        layers = layers,
        initializer = initializer,
        activation = activation,
        optimizer = optimizer,
        learning_rate = lr * 0.0001,
        metric = [],
        num_domain = num_domain,
        num_boundary = num_boundary,
        num_test = num_test,
        iters = iters)
    
    #sol = pinn.exact_solution()
    choice = st.radio(" ",
                ["PINN Solution","Paramter Identification"],
                captions = [],
                )
        
    if choice == "PINN Solution":
        if st.button('PINN Solve'):
            x, t, y =  pinn.solve()           
            X, T = np.meshgrid(x, t)
            
            fig, ax  = plt.subplots(1,2 , figsize = (14,7))
            
            im = ax[0].contourf(T, X, y, cmap="coolwarm", levels=100)
            ax[0].set_xlabel('Time')
            ax[0].set_ylabel('X')
            ax[0].set_title('PINNS solution W(T,X)')  
            fig.colorbar(im, ax = ax[0])
            
            im = ax[1].contourf(T, X, pinn.exact_solution(T,X), cmap="coolwarm", levels=100)
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('X')
            ax[1].set_title('Exact solution W(T,X)')
            fig.colorbar(im, ax = ax[1])

            st.pyplot(fig)
    if choice == "Paramter Identification":
     
        st.text_area(' ', 
                     "In this part we will simulate an environment with background values for parameters. Suppose an experimenter collected some date points, and wishes to find the background paramters values.")
       
        param_star = st.number_input(label="What is the background parameter?",
                                     min_value = 0.1,
                                     max_value = 2000.,
                                     value = 1. )
        num_pts = st.number_input(label = "How many data points are collected?",
                                  min_value = 1,
                                  max_value = 2000,
                                  value = 100
                                  )
        noise_amp = st.number_input(label = 'If you want to creat a noisy environment, sepcify its amplitude',
                                    min_value = 0.,
                                    max_value = 1.,
                                    value = 0.) 
        
        
        if st.button('PINN Param Identification'): 
            
            a_optim =  pinn.find_param(param_star, num_pts, noise_amp)
            st.text_area('Result:',
                         f'The identified value for background paramter is { a_optim[0] }')
            
elif prob == 'Linear System of PDEs':
    st.image("ProblemSets/LinearSystemPDE.png", caption = "Problem explanation, ref: master notebook")
        
    alpha = st.number_input(label = 'alpha (compatible with boundary = -1)',
                            min_value = -20.,
                            max_value = 20.,
                            value = -1.)
    beta = st.number_input(label = 'beta (compatible with boundary = 0)',
                            min_value = -20.,
                            max_value = 20.,
                            value = 0.)
    gamma = st.number_input(label = 'gamma, (compatible with boundary = 1)',
                            min_value = -20.,
                            max_value = 20.,
                            value = 1.)
    delta = st.number_input(label = 'delta, (compatible with boundary = -2)',
                            min_value = -20.,
                            max_value = 20.,
                            value = -2.)
   
    initializer = st.selectbox("Choose the kernel_initializer",
                               ('He normal', 'He uniform', 'Glorot normal', 'Glorot uniform'))
    activation = st.selectbox('Choose the activation function',
                              ('tanh', 'elu', 'relu', 'sigmoid', 'sin'))
    optimizer = st.selectbox('Choose the optimizer',
                            ('adam', "sgd", "sgdnesterov", "adagrad", ))
   
    
    
    layers_t = st.text_input("Enter the Neural Network shape (a sequence separated by commas):",
                             "2,30,30,2")
    layers_s = layers_t.split(",")
    layers = [int(l) for l in layers_s]
    
    lr = st.slider(label = "Learning rate (in unit of 0.0001)",
                   min_value = 1.,
                   max_value = 1000.,
                   value = 10.)
    
    num_domain = st.number_input('Number of the points in the domain',
                                 min_value = 10,
                                 max_value= 10000,
                                 value = 1000)
    num_boundary = st.number_input('Number of the boundary points',
                                   min_value = 0,
                                   max_value = 1000,
                                   value = 1
                                   )
    num_test = st.number_input('How many test points?',
                               min_value=1,
                               max_value=1000,
                               value =100
                               )
    iters = st.number_input('How many iterations?',
                            min_value=100,
                            max_value=10000,
                            value = 1000)
    

    
    pinn = LinearSystem.PDE_PINN(
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        delta = delta,
        layers = layers,
        initializer = initializer,
        activation = activation,
        optimizer = optimizer,
        learning_rate = lr * 0.0001,
        metric = [],
        num_domain = num_domain,
        num_boundary = num_boundary,
        num_test = num_test,
        iters = iters)
    
    choice = st.radio(" ",
                ["PINN Solution","Paramter Identification"],
                captions = [],
                )
    
    if choice == "PINN Solution":
        if st.button('PINN Solve'):
            x, t, Z_pred =  pinn.solve()           
            X, T = np.meshgrid(x, t)
            
            fig, ax  = plt.subplots(2,2 , figsize = (14,14))
            
            im = ax[0,0].contourf(T, X,  Z_pred[:,:,0], cmap="coolwarm", levels=100)
            ax[0,0].set_xlabel('Time')
            ax[0,0].set_ylabel('X')
            ax[0,0].set_title('PINNS solution v(T,X)')  
            fig.colorbar(im, ax = ax[0,0])
            
            im = ax[0,1].contourf(T, X, pinn.v_sol(X,T), cmap="coolwarm", levels=100)
            ax[0,1].set_xlabel('Time')
            ax[0,1].set_ylabel('X')
            ax[0,1].set_title('Exact solution v(T,X)')
            fig.colorbar(im, ax = ax[0,1])
            
            im = ax[1,0].contourf(T, X,  Z_pred[:,:,1], cmap="coolwarm", levels=100)
            ax[1,0].set_xlabel('Time')
            ax[1,0].set_ylabel('X')
            ax[1,0].set_title('PINNS solution w(T,X)')  
            fig.colorbar(im, ax = ax[1,0])
            
            im = ax[1,1].contourf(T, X, pinn.w_sol(X,T), cmap="coolwarm", levels=100)
            ax[1,1].set_xlabel('Time')
            ax[1,1].set_ylabel('X')
            ax[1,1].set_title('Exact solution w(T,X)')
            fig.colorbar(im, ax = ax[1,1])

            st.pyplot(fig)
    
    if choice == "Paramter Identification":
     
        st.text_area(' ', 
                     "In this part we will simulate an environment with background values for parameters. Suppose an experimenter collected some date points, and wishes to find the background paramters values.")
       
        st.text_area(' ',
                     'You can change the initial values of the paramters. The background values of the parameters which are compatible with the boundary condition, and PINN should find them are: alpha = -1, beta = 0, gamma = 1, delta = -2.')
        num_pts = st.number_input(label = "How many data points are collected?",
                                  min_value = 1,
                                  max_value = 2000,
                                  value = 100
                                  )
        noise_amp = st.number_input(label = 'If you want to creat a noisy environment, sepcify its amplitude',
                                    min_value = 0.,
                                    max_value = 1.,
                                    value = 0.) 
        
        
        if st.button('PINN Param Identification'): 
            
            par_optim =  pinn.find_param(num_pts, noise_amp)
            st.text_area('The identified values for background parameters are :',
                         f'alpha =  { par_optim[0] }, beta = {par_optim[1]}, gamma = {par_optim[2]}, delta = {par_optim[3]}')
            
elif prob == 'Hamilton Jacobi':
    st.image("ProblemSets/Hamilton_Jacobi2.png", caption = "Problem explanation, ref: master notebook")
        
    a = st.number_input(label = 'a',
                            min_value = -20.,
                            max_value = 20.,
                            value = 1.)
    b = st.number_input(label = 'b',
                            min_value = -20.,
                            max_value = 20.,
                            value = 1.)
    
    
    initializer = st.selectbox("Choose the kernel_initializer",
                               ('He normal', 'He uniform', 'Glorot normal', 'Glorot uniform'))
    activation = st.selectbox('Choose the activation function',
                              ('tanh', 'elu', 'relu', 'sigmoid', 'sin'))
    optimizer = st.selectbox('Choose the optimizer',
                            ('adam', "sgd", "sgdnesterov", "adagrad", ))
   
    
    
    layers_t = st.text_input("Enter the Neural Network shape (a sequence separated by commas):",
                             "2,30,30,1")
    layers_s = layers_t.split(",")
    layers = [int(l) for l in layers_s]
    
    lr = st.slider(label = "Learning rate (in unit of 0.0001)",
                   min_value = 1.,
                   max_value = 1000.,
                   value = 10.)
    
    num_domain = st.number_input('Number of the points in the domain',
                                 min_value = 10,
                                 max_value= 10000,
                                 value = 1000)
    num_boundary = st.number_input('Number of the boundary points',
                                   min_value = 0,
                                   max_value = 1000,
                                   value = 1
                                   )
    num_test = st.number_input('How many test points?',
                               min_value=1,
                               max_value=1000,
                               value =100
                               )
    iters = st.number_input('How many iterations?',
                            min_value=100,
                            max_value=10000,
                            value = 1000)
    

    
    pinn = Hamilton_Jacobi.HJ_PINN(
        a = a,
        b = b,
        layers = layers,
        initializer = initializer,
        activation = activation,
        optimizer = optimizer,
        learning_rate = lr * 0.0001,
        metric = [],
        num_domain = num_domain,
        num_boundary = num_boundary,
        num_test = num_test,
        iters = iters)
    
    choice = st.radio(" ",
                ["PINN Solution","Paramter Identification"],
                captions = [],
                )
    
        
    if choice == "PINN Solution":
        if st.button('PINN Solve'):
            x, t, y =  pinn.solve()           
            X, T = np.meshgrid(x, t)
            
            fig, ax  = plt.subplots(1,2 , figsize = (14,7))
            
            ax[0].contourf(X, T, y[:,:,0], cmap="jet", levels=100)
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('t')
            ax[0].set_title('PINN solution')  
           
            
            ax[1].contourf(X, T,  np.sin(2*np.arctan(np.exp(-T)*np.tan(X/2))), cmap="jet", levels=100)
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('t')
            ax[1].set_title('Exact solution')
            

            st.pyplot(fig)
    
    if choice == "Paramter Identification":
     
        st.text_area(' ', 
                     "In this part we will simulate an environment with background values for parameters. Suppose an experimenter collected some date points, and wishes to find the background paramters values.")
       
        st.text_area(' ',
                     'You can change the initial values of the paramters. The background values of the parameters which are compatible with the boundary condition, and PINN should find them are: a = 1, b = 1')
        num_pts = st.number_input(label = "How many data points are collected?",
                                  min_value = 1,
                                  max_value = 2000,
                                  value = 100
                                  )
        noise_amp = st.number_input(label = 'If you want to creat a noisy environment, sepcify its amplitude',
                                    min_value = 0.,
                                    max_value = 1.,
                                    value = 0.) 
        
        
        if st.button('PINN Param Identification'): 
            
            par_optim =  pinn.find_param(num_pts, noise_amp)
            st.text_area('The identified values for background parameters are :',
                         f'a =  { par_optim[0] }, b = {par_optim[1]}')
            
elif prob == 'Klein Gordon':
    
    Example = st.radio("There are two types of initial conditions that are considered here",
                ["Case1",'Case2'],
                captions = [],
                )
    
    if Example == 'Case1':
        st.image("ProblemSets/Klein-Gordon1.png", caption = "Problem explanation, ref: master notebook")
    elif Example == 'Case2':
        st.image("ProblemSets/Klein-Gordon2.png", caption = "Problem explanation, ref: master notebook")
      
    a = st.number_input(label = 'a',
                            min_value = -20.,
                            max_value = 20.,
                            value = 1.)
    b = st.number_input(label = 'b',
                            min_value = -20.,
                            max_value = 20.,
                            value = 1.)
    
    t_max = st.number_input(label = 'Size of the temporal dimension',
                            min_value = 2.,
                            max_value = 20.,
                            value = 4.)
    
    initializer = st.selectbox("Choose the kernel_initializer",
                               ('He normal', 'He uniform', 'Glorot normal', 'Glorot uniform'))
    activation = st.selectbox('Choose the activation function',
                              ('tanh', 'elu', 'relu', 'sigmoid', 'sin'))
    optimizer = st.selectbox('Choose the optimizer',
                            ('adam', "sgd", "sgdnesterov", "adagrad", ))
   
    
    
    layers_t = st.text_input("Enter the Neural Network shape (a sequence separated by commas):",
                             "2,30,30,1")
    layers_s = layers_t.split(",")
    layers = [int(l) for l in layers_s]
    
    lr = st.slider(label = "Learning rate (in unit of 0.0001)",
                   min_value = 1.,
                   max_value = 1000.,
                   value = 10.)
    
    num_domain = st.number_input('Number of the points in the domain',
                                 min_value = 10,
                                 max_value= 10000,
                                 value = 1000)
    num_boundary = st.number_input('Number of the boundary points',
                                   min_value = 0,
                                   max_value = 1000,
                                   value = 1
                                   )
    num_test = st.number_input('How many test points?',
                               min_value=1,
                               max_value=1000,
                               value =100
                               )
    iters = st.number_input('How many iterations?',
                            min_value=100,
                            max_value=10000,
                            value = 1000)
    

    
    pinn = Klein_Gordon.KG_PINN(
        a = a,
        b = b,
        t_max=t_max,
        case = Example,
        layers = layers,
        initializer = initializer,
        activation = activation,
        optimizer = optimizer,
        learning_rate = lr * 0.0001,
        metric = [],
        num_domain = num_domain,
        num_boundary = num_boundary,
        num_test = num_test,
        iters = iters)
    
    choice = st.radio(" ",
                ["PINN Solution","Paramter Identification"],
                captions = [],
                )
    if choice == 'PINN Solution':
        if st.button('PINN Solve'):
            x, t, y =  pinn.solve()           
            X, T = np.meshgrid(x, t)
            
            fig, ax  = plt.subplots(1,2 , figsize = (14,7))
            
            ax[0].contourf(X, T, y[:,:,0], cmap="seismic", levels=100)
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('t')
            ax[0].set_title('PINN solution')  
        
        
            ax[1].contourf(X, T,  pinn.exact_sol(X,T), cmap="seismic", levels=100)
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('t')
            ax[1].set_title('Exact solution')
        

            st.pyplot(fig)
    if choice == 'Paramter Identification':
        
        st.text_area(' ', 
                     "In this part we will simulate an environment with background values for parameters. Suppose an experimenter collected some date points, and wishes to find the background paramters values.")
       
        st.text_area(' ',
                     f'You can change the initial values of the paramters. The background values of the parameters which are compatible with the boundary condition, and PINN should find them are: a = {a}, b = {b}')
        num_pts = st.number_input(label = "How many data points are collected?",
                                  min_value = 1,
                                  max_value = 2000,
                                  value = 100
                                  )
        noise_amp = st.number_input(label = 'If you want to creat a noisy environment, sepcify its amplitude',
                                    min_value = 0.,
                                    max_value = 1.,
                                    value = 0.) 
        
        
        if st.button('PINN Param Identification'): 
            
            par_optim =  pinn.find_param(num_pts, noise_amp)
            st.text_area('The identified values for background parameters are :',
                         f'a =  { par_optim[0] }, b = {par_optim[1]}')
    
    
    
    