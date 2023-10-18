import torch
from config import *
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

# --- Modified version of CCA loss function originally introduced by Galen Andrew et al. (2013) 
# and re-implemented by https://github.com/Michaelvll/DeepCCA

class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()

        o1 =  H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)



        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        # Workaround !!! USE BATCH > 16
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        
        # Newest way but not debugged
        # [D1, V1] = torch.linalg.eig(SigmaHat11)
        # [D2, V2] = torch.linalg.eig(SigmaHat22)
        
        # Original
        # [D1, V1] = torch.eig(SigmaHat11, eigenvectors=True)
        # [D2, V2] = torch.eig(SigmaHat22, eigenvectors=True)

        D1 = D1.unsqueeze(1)
        posInd1 = torch.gt(D1[:, 0], eps).nonzero()[:, 0]
        D1 = D1[posInd1, 0]
        V1 = V1[:, posInd1]
        D1 = torch.squeeze(D1)  # Remove extra dimensions from D1
        # Reshape D1 to 1-dimensional tensor
        D2 = D2.unsqueeze(1)
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        posInd2 = torch.gt(D2[:, 0], eps).nonzero()[:, 0]
        D2 = D2[posInd2, 0]
        V2 = V2[:, posInd2]
        D2 = torch.squeeze(D2)
        
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)


        if self.use_all_singular_values:
    
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
  
        else:
   
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) 
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).float()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

cca = cca_loss(outdim_size, use_all_singular_values, device = device)


def cal_loss(label, args, pred=None, text_embed=None, eeg_embed=None):
    loss = None
    n_correct = None
    
    loss = args.ce_weight * F.cross_entropy(pred, label, reduction='sum')
    pred = pred.max(1)[1]
    n_correct = pred.eq(label).sum().item()
    
    if args.loss == 'CE':
        return loss, n_correct
    elif args.loss == 'CCA' and args.modality == 'fusion':
        loss = loss + args.cca_weight * cca.loss(text_embed, eeg_embed)
        return loss, n_correct
    elif args.loss == 'WD' and args.modality == 'fusion':
        loss = loss + args.wd_weight * torch.tensor(wasserstein_distance(text_embed.cpu().detach().numpy().flatten(), eeg_embed.cpu().detach().numpy().flatten()), requires_grad=True)
        return loss, n_correct
    elif args.loss == 'CCAWD' and args.modality == 'fusion':
        loss = loss + args.cca_weight * cca.loss(text_embed, eeg_embed)
        loss = loss + args.wd_weight * torch.tensor(wasserstein_distance(text_embed.cpu().detach().numpy().flatten(), eeg_embed.cpu().detach().numpy().flatten()), requires_grad=True)
        return loss, n_correct    