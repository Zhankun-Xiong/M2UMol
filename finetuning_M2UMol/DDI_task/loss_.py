import torch
from torch import nn
import torch.nn.functional as F
#from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss

class NtXent(nn.modules.loss._Loss):
    def __init__(self,temperature, return_logits=False):
        super(NtXent, self).__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        #z_i=z_i.cpu().detach()
        #z_j=z_j.cpu().detach()
        N = len(z_i)
        z_i = F.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = F.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = F.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = F.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs
        #print((loss_i + loss_j))
        return loss_i + loss_j
# class NTXentAE(_Loss):
#     '''
#         Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
#         Args:
#             z1, z2: Tensor of shape [batch_size, z_dim]
#             tau: Float. Usually in (0,1].
#             norm: Boolean. Whether to apply normlization.
#         '''
#
#     def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0, reconstruction_reg = 1) -> None:
#         super(NTXentAE, self).__init__()
#         self.norm = norm
#         self.tau = tau
#         self.uniformity_reg = uniformity_reg
#         self.variance_reg = variance_reg
#         self.covariance_reg = covariance_reg
#         self.mse_loss = MSELoss()
#         self.reconstruction_reg = reconstruction_reg
#
#     def forward(self, z1, z2, **kwargs): #, distances, distance_pred
#         batch_size, _ = z1.size()
#         sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
#
#         if self.norm:
#             z1_abs = z1.norm(dim=1)
#             z2_abs = z2.norm(dim=1)
#             sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)
#
#         sim_matrix = torch.exp(sim_matrix / self.tau)
#         pos_sim = torch.diagonal(sim_matrix)
#         loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
#         # print(111111111111111111)
#         # print(sim_matrix[0:10])
#         # print(pos_sim)
#         # print((sim_matrix.sum(dim=1)))
#         # print((sim_matrix.sum(dim=1) - pos_sim))
#         # print(loss)
#         loss = - torch.log(loss).mean()
#
#         # if self.variance_reg > 0:
#         #     loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
#         # if self.covariance_reg > 0:
#         #     loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
#         # if self.uniformity_reg > 0:
#         #     loss += self.uniformity_reg * uniformity_loss(z1, z2)
#         return loss#, self.reconstruction_reg * self.mse_loss(distances, distance_pred)