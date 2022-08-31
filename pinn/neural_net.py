#from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from pinn.util import log
from pathlib import Path
import pinn.get_points
import time

class MLP(nn.Module):
    
    # Define the MLP

    def __init__(
        self, pars, device
    ) -> None:

        super().__init__()

        # Add number of MLP input and outputs to the layers list
        layers = [3,*pars['layers'],2]
        
        # Built the MLP
        modules = []
        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(ResidualBlock(_out))
        
        # Remove last block
        modules.pop()

        self.model = nn.Sequential(*modules)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # Forward pass
        Y_n = self.model(X)
        Y_p = self.particular_solution(X)
        D = self.boundary_distance(X)

        return D * Y_n + (1-D) * Y_p

    def particular_solution(self,X):
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)

        u = torch.sin(2*np.pi*x)*torch.sin(2*np.pi*y)
        v = torch.sin(np.pi*x)*torch.sin(np.pi*y)

        return torch.hstack((u,v))

    def boundary_distance(self,X):

        alpha = 26.4 # Reaches 0.99 at t = 0.1
        #alpha = 10.56 # Reaches 0.99 at t = 0.25

        t = X[:,0].reshape(-1, 1)
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)

        dt = torch.tanh(t*alpha)
        dx = 4*x*(1-x)
        dy = 4*y*(1-y)

        return torch.hstack((dt*dx*dy,dt*dx*dy))

class ResidualBlock(nn.Module):

    # Define a block with two layers and a residual connection
    def __init__(self,_size:int):
        super().__init__()
        self.Layer1 = nn.Tanh()
        self.Linear = nn.Linear(_size, _size)
        self.Layer2 = nn.Tanh()

    def forward(self,x):
        return x + self.Layer2(self.Linear(self.Layer1(x)))

class PINN:
    def __init__(self, nf, ns, pars: dict, device: torch.device = 'cpu') -> None:

        # Parameters
        self.pars = pars
        self.device = device
        self.nf = nf
        self.ns = ns
        self.nu = pars['nu']

        self.ls_f1 = torch.tensor(0).to(self.device)
        self.ls_f2 = torch.tensor(0).to(self.device)
        self.ls_f = torch.tensor(0).to(self.device)
        self.ls_s = torch.tensor(0).to(self.device)

        # Sample points
        self.sample_points()
        self.zeros = torch.zeros(self.X_f.shape).to(self.device)

        [X_s,Y_s] = pinn.get_points.solution(ns,pars)
        self.X_s = torch.tensor(X_s,dtype=torch.float,requires_grad=True).to(self.device)
        self.Y_s = torch.tensor(Y_s,dtype=torch.float,requires_grad=False).to(self.device)

        # Initialize Network
        self.net = MLP(pars,device)
        self.net = self.net.to(device)

        if pars['loss_type'] == 'l1':
            self.loss = nn.L1Loss().to(device)
        elif pars['loss_type'] == 'mse':
            self.loss = nn.MSELoss().to(device)

        self.min_ls_tol = 0.01
        self.min_ls_wait = 10000
        self.min_ls_window = 1000

        self.start_time = time.time()

        self.ls = 0
        self.iter = 0

        self.ls_hist = np.zeros((pars['epochs'],4))

        # Optimizer parameters
        if pars['opt_method'] == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=pars['opt_lr'])
        elif pars['opt_method'] == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.net.parameters(),lr=pars['opt_lr'])
        elif pars['opt_method'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=pars['opt_lr'])
        else:
            raise Exception("Unknown optimization method")

    def sample_points(self):
        X_f = pinn.get_points.domain(self.nf,self.pars)
        self.X_f = torch.tensor(X_f,dtype=torch.float,requires_grad=True).to(self.device)

    def eq_loss(self, X: torch.Tensor):

        # Forward pass
        t = X[:,0].reshape(-1, 1)
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)
        Y = self.net(torch.hstack((t,x,y)))
        
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
        R1 = u_t + u*u_x + v*u_y - self.nu*(u_xx + u_yy)
        R2 = v_t + u*v_x + v*v_y - self.nu*(v_xx + v_yy)

        self.ls_f1 = self.loss(R1,torch.zeros_like(R1))
        self.ls_f2 = self.loss(R2,torch.zeros_like(R1))

        return self.ls_f1 + self.ls_f2

    def sample_loss(self, X: torch.Tensor, Y_gt: torch.Tensor):

        if self.ns == 0:
            return 0

        Y_pred = self.net(X)
        return self.loss(Y_pred,Y_gt)

    def closure(self) -> torch.nn:
        
        if self.ns > 0:
            self.ls_s = self.sample_loss(self.X_s,self.Y_s)
        if self.nf > 0:
            self.ls_f = self.eq_loss(self.X_f)

        if self.ns > 0 and self.nf > 0:
            self.ls = self.ls_s + self.ls_f
        elif self.ns > 0:
            self.ls = self.ls_s
        elif self.nf > 0:
            self.ls = self.ls_f

        self.optimizer.zero_grad()
        self.ls.backward()

        return self.ls

    def stopping(self):
        # Stop the training if the median loss of the last min_ls_window steps is not improved in min_ls_wait by a factor of min_ls_tol
        if self.iter > self.min_ls_wait + self.min_ls_window:

            old_list = sorted(self.ls_hist[self.iter-self.min_ls_wait-self.min_ls_window+1:self.iter-self.min_ls_wait,0])
            new_list = sorted(self.ls_hist[self.iter-self.min_ls_window+1:self.iter,0])
            median_ind = self.min_ls_window//2

            if new_list[median_ind] > old_list[median_ind] * (1-self.min_ls_tol):
                return True

        return False

    def train(self):
        self.net.train()

        for i in range(1,self.pars['epochs']):
            self.iter += 1

            if self.pars['shuffle'] and i%self.pars['shuffle']==0:
                self.sample_points()

            try:
                self.optimizer.step(self.closure)
            except KeyboardInterrupt:
                print("Stopped by user")
                self.save(0)
                try:
                    input('Press Enter to resume or Ctrl+C again to stop')
                except KeyboardInterrupt:
                    break
                

            log.info(f'Epoch: {self.iter}, Loss: {self.ls:.3e}, Loss_F: {self.ls_f:.3e} ({self.ls_f1:.3e} + {self.ls_f2:.3e}), Loss_S: {self.ls_s:.3e}')

            self.ls_hist[i,:] = torch.hstack((self.ls,self.ls_f1,self.ls_f2,self.ls_s)).cpu().detach().numpy()

            # Save the model every n steps
            #if i%10000==0:
            #    self.save(i)

            # Stop if early stopping criterium is met
            #if self.stopping():
            #    break

    def save(self,iter):
        Path(str(self.pars['save_path'].parents[0])).mkdir(parents=True, exist_ok=True)
        if iter == -1:
            save_path = self.pars['save_path']
        elif iter == 0:
            save_path = "{0}_partial.{1}".format(*str(self.pars['save_path']).rsplit('.', 1))
        else:
            save_path = f"{str(self.pars['save_path'])[0:-3]}_iter{iter}.pt"

        log.info(f'Saving model to {save_path}')

        ls_hist_temp = self.ls_hist[0:np.nonzero(self.ls_hist[:,0])[0][-1],:]

        torch.save({'model': self.net.state_dict(),'pars':self.pars,'loss':ls_hist_temp, 'time':time.time()-self.start_time, 'memory':torch.cuda.max_memory_allocated(self.device)/(1024*1024)}, save_path)