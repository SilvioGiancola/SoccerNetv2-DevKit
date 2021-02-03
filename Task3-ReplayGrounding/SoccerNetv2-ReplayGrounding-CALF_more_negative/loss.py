import torch


####################################################################################################################################################

# Context-aware loss function

####################################################################################################################################################

import torch

class SymmetricContextAwareLoss(torch.nn.Module):

    def __init__(self, K, receptive_field, hit_radius = 0.1, miss_radius = 0.9):

        super(SymmetricContextAwareLoss,self).__init__()

        self.K = K
        self.hit_radius = float(hit_radius)
        self.miss_radius = float(miss_radius)
        self.receptive_field = receptive_field

    def forward(self, gt_label, pred_score):
    
        zeros = torch.zeros(pred_score.size()).to(pred_score.device).type(torch.float)
        pred_score = torch.clamp(pred_score, 0.001, 0.999)
        
        case1 = self.DownStep(gt_label, self.K[0]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + self.miss_radius))
        case2 = self.Interval(gt_label, self.K[0], self.K[1]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(self.PartialIdentity(gt_label,self.K[0],self.K[1])-self.K[0])/(self.K[1]-self.K[0])) + torch.log(zeros + self.miss_radius))
        #note cases 1 and 2 could be regrouped with DownStep(gt_label, self.K[1]) * formula loss of case2
        case3 = self.Interval(gt_label, self.K[1], 0.) * torch.max(zeros, - torch.log(1.-pred_score + pred_score*(self.PartialIdentity(gt_label,self.K[1],0.)-0.)/(self.K[1]-0.)) + torch.log(zeros + 1.-self.hit_radius))
        case4 = self.Interval(gt_label, 0., self.K[2]) * torch.max(zeros, - torch.log(1.-pred_score + pred_score*(self.PartialIdentity(gt_label,0.,self.K[2])-0.)/(self.K[2]-0.)) + torch.log(zeros + 1.-self.hit_radius))
        case5 = self.Interval(gt_label, self.K[2], self.K[3]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(self.PartialIdentity(gt_label,self.K[2],self.K[3])-self.K[3])/(self.K[2]-self.K[3])) + torch.log(zeros + self.miss_radius))
        case6 = self.UpStep(gt_label, self.K[3]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + self.miss_radius))
        #note cases 5 and 6 could be regrouped 
        
        L = case1 + case2 + case3 + case4 + case5 + case6
        
        L[:,0:self.receptive_field//2,:] *= 0.
        L[:,-self.receptive_field//2:,:] *= 0.
        
        return torch.sum(L)

    def UpStep(self,x,a): #0 if x<a, 1 if x >= a

        return 1.-torch.max(0.*x,torch.sign(a-x))

    def DownStep(self,x,a): #1 if x < a, 0 if x >=a

        return torch.max(0.*x,torch.sign(a-x))

    def Interval(self,x,a,b): # 1 if a<= x < b, 0 otherwise
        
        return self.UpStep(x,a) * self.DownStep(x,b)

    def PartialIdentity(self,x,a,b):#a if x<a, x if a<= x <b, b if x >= b

        return torch.min(torch.max(x,0.*x+a),0.*x+b)




####################################################################################################################################################

# Spotting loss

####################################################################################################################################################

class ReplayGroundingSpottingLoss(torch.nn.Module):

    def __init__(self, lambda_coord, lambda_noobj):
        super(ReplayGroundingSpottingLoss,self).__init__()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def forward(self,y_true, y_pred):

        loss = torch.sum(y_true[:,:,0]*self.lambda_coord*torch.square(y_true[:,:,1]-y_pred[:,:,1])  +  y_true[:,:,0]*torch.square(y_true[:,:,0]-y_pred[:,:,0]) +  (1-y_true[:,:,0])*self.lambda_noobj*torch.square(y_true[:,:,0]-y_pred[:,:,0])) #-y_true[:,:,0]*torch.sum(y_true[:,:,2:]*torch.log(y_pred[:,:,2:]),axis=-1)
        return loss
