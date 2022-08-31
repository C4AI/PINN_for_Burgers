import torch
import numpy as np
from pinn.neural_net import MLP
from pinn.util import log
from scipy.io import loadmat
from scipy.interpolate import interpn

def get_pars(model_path):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['pars']
    
    return pars

def get_loss(model_path):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['loss']
    
    return pars

def evaluate(t,x,y,model_path,device='cpu'):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['pars']
    model = MLP(pars,device)
    model.load_state_dict(model_file['model'])
    model.eval()

    model = model.to(device)

    # Define grid
    [x_grid, t_grid, y_grid] = np.meshgrid(x,t,y)
    X = np.hstack((t_grid.flatten()[:,None],x_grid.flatten()[:,None],y_grid.flatten()[:,None]))
    X = torch.tensor(X,dtype=torch.float).to(device)

    # Evaluate model
    Y_pred = model(X)

    Y_pred = Y_pred.cpu().detach().numpy()

    u_pred = Y_pred[:,0].reshape(t_grid.shape)
    v_pred = Y_pred[:,1].reshape(t_grid.shape)

    return u_pred, v_pred, t_grid, x_grid, y_grid, X

def get_residuals(t,x,y,model_path,device='cpu'):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['pars']
    model = MLP(pars,device)
    model.load_state_dict(model_file['model'])
    model.eval()

    model = model.to(device)

    # Define grid
    [x_grid, t_grid, y_grid] = np.meshgrid(x,t,y)
    X_np = np.hstack((t_grid.flatten()[:,None],x_grid.flatten()[:,None],y_grid.flatten()[:,None]))

    X = torch.tensor(X_np,dtype=torch.float,requires_grad=True).to(device)

    t = X[:,0].reshape(-1, 1)
    x = X[:,1].reshape(-1, 1)
    y = X[:,2].reshape(-1, 1)

    # Get residuals
    # Forward pass
    t = X[:,0].reshape(-1, 1)
    x = X[:,1].reshape(-1, 1)
    y = X[:,2].reshape(-1, 1)
    Y = model(torch.hstack((t,x,y)))
    
    u = Y[:,0].reshape(-1, 1)
    v = Y[:,1].reshape(-1, 1)

    # Get derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]

    # Compute residuals
    R1 = u_t + u*u_x + v*u_y - pars['nu']*(u_xx + u_yy)
    R2 = v_t + u*v_x + v*v_y - pars['nu']*(v_xx + v_yy)

    #R1 = e_t
    #R2 = ud_x
    #R3 = vd_y

    R1 = R1.cpu().detach().numpy()
    R1 = R1.reshape(t_grid.shape)
    R2 = R2.cpu().detach().numpy()
    R2 = R2.reshape(t_grid.shape)

    return R1, R2, t_grid, x_grid, y_grid, X

def load_data(t,x,y,save_path):

    [x_grid, t_grid, y_grid] = np.meshgrid(x,t,y)
    X = np.hstack((t_grid.flatten()[:,None],x_grid.flatten()[:,None],y_grid.flatten()[:,None]))

    data = loadmat(save_path)
    t_data = data['t'].flatten()
    x_data = data['x'].flatten()
    y_data = data['y'].flatten()

    X_data = (t_data,x_data,y_data)
    U_data = data['uref'].transpose((2,1,0))
    V_data = data['vref'].transpose((2,1,0))
    
    u_int = interpn(X_data,U_data,X).reshape(t_grid.shape)
    v_int = interpn(X_data,V_data,X).reshape(t_grid.shape)

    return u_int, v_int
    