import torch
import numpy as np
import argparse
import time
from pinn.neural_net import PINN
from pinn.util import log
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        prog='Training step',
        usage='%(prog)s [options] parser',
        description='Parser for hyperparams training')
    
    parser.add_argument('--path',
                        type=str,
                        default='',
                        help='Use to manually select the model file name')

    parser.add_argument('--solution',
                        type=str,
                        default='reference_solution.mat',
                        help='File containing the solution via simulation')

    parser.add_argument('--comment',
                        type=str,
                        default='',
                        help='String to be added to the end of the automatically generated file name')

    parser.add_argument('--folder',
                        type=str,
                        default='models',
                        help='Folder where the automatically named model will be saved')   

    parser.add_argument('--resume',
                        type=str,
                        default='',
                        help='Model to be used as initial guess')

    parser.add_argument('--nf',
                        type=int,
                        default=10000,
                        help='Number of function evaluation points')

    parser.add_argument('--ns',
                        type=int,
                        default=1000,
                        help='Number of solution points')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=100000,
                        help='Number of epochs for training')

    parser.add_argument('--nlayers',
                        type=int,
                        default=2,
                        help='MLP number residual blocks')

    parser.add_argument('--nneurons',
                        type=int,
                        default=20,
                        help='MLP number neurons per layer')

    parser.add_argument('--shuffle',
                        type=int,
                        default=0,
                        help='Reshuffle sample every n itertions - 0 for fixed sample')

    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed')                

    parser.add_argument('--opt_method',
                        type=str,
                        default='adam',
                        help='Optimization algorithm (adam, lbfgs or sgd)')

    parser.add_argument('--opt_lr',
                        type=float,
                        default=1e-2,
                        help='Learning rate for the optimization algorithm')

    parser.add_argument('--loss',
                        type=str,
                        default='l1',
                        help='Type of reduction to be used for each loss (l1 or mse)')

    parser.add_argument('--dev',
                        type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device to run the model')
    
    args = parser.parse_args()
    
    return args

def main():

    # Define parameters
    pars = dict()
    pars['xi'] = 0
    pars['xf'] = 1
    pars['yi'] = 0
    pars['yf'] = 1
    pars['tf'] = 1
    pars['tf'] = 1
    pars['nu'] = 0.01/np.pi

    # Retrive arguments
    args = get_args()
    nf = args.nf
    ns = args.ns
    pars['epochs'] = args.epochs
    pars['shuffle'] = args.shuffle
    device = args.dev
    resume = args.resume
    pars['solution_file'] = args.solution
    

    pars['opt_method'] = args.opt_method
    pars['opt_lr'] = args.opt_lr
    pars['loss_type'] = args.loss

    pars['layers'] = [args.nneurons for i in range(0,args.nlayers)]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.path) == 0:
        pars['save_path'] = Path(f'{args.folder}/model_nf{nf}_ns{ns}_MLPRes_2x{args.nlayers}x{args.nneurons}_shuffle{args.shuffle}_seed{args.seed}_{args.opt_method}_lr{args.opt_lr}_loss_{args.loss}{args.comment}.pt')
    else:
        pars['save_path'] = Path(args.path)

    if pars['save_path'].is_file():
        return

    log.info(f'Model will be saved to: {pars["save_path"]}')
    log.info(f'Number of samples - Function evaluation: {nf}, Solution samples: {ns}')
    log.info(f'Using device: {device}')

    # Train model
    model = PINN(nf, ns, pars, device)
    if len(resume) != 0:
        resume_file = torch.load(resume)
        model.net.load_state_dict(resume_file['model'])

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time

    log.info(f'Training time: {elapsed:.4f}s')
    model.save(-1)
    log.info('Finished training.')

if __name__== "__main__":
    main()