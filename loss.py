from typing import List

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torch
import torch.nn as nn

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection =torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)

    return loss

class TowerLoss(nn.Module):
    def __init__(self):
        super(TowerLoss, self).__init__()
        # todo add device handling

    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta

        return torch.mean(L_g * y_true_cls * training_mask) + classification_loss


class PODSpacialLoss(nn.Module):
    def __init__(self, dim: int):
        super(PODSpacialLoss, self).__init__()
        self.dim = dim
        # todo add device handling

    def forward(self,
                teacher_featuremaps: torch.Tensor,
                student_featuremaps: torch.Tensor) -> torch.Tensor:

        loss = torch.tensor(0.) # todo add device handling

        for a, b in zip(teacher_featuremaps, student_featuremaps):
            a_pooled = a.sum(dim=self.dim).view(a.shape[0], -1)
            b_pooled = b.sum(dim=self.dim).view(b.shape[0], -1)

            distance = torch.mean(torch.frobenius_norm(a_pooled-b_pooled, dim=-1))
            loss += distance

        return loss


class PODFlatLoss(nn.Module):
    def __init__(self):
        super(PODFlatLoss, self).__init__()
        # todo add device handling

    def forward(self,
                teacher_logits: torch.Tensor,
                student_logits: torch.Tensor) -> torch.Tensor:
        flattened_teacher_logits = torch.flatten(teacher_logits, start_dim=1)
        flattened_student_logits = torch.flatten(student_logits, start_dim=1)
        return F.l1_loss(flattened_student_logits, flattened_teacher_logits)



class PODLoss(nn.Module):

    def __init__(self, height_coef: float = 1., width_coef: float = 1., flat_coef: float = 1.):
        super(PODLoss, self).__init__()
        self.spacial_height_loss = PODSpacialLoss(dim=1)
        self.spacial_width_loss = PODSpacialLoss(dim=2)
        self.pod_flat_loss = PODFlatLoss()

        self.height_coef = height_coef
        self.width_coef = width_coef
        self.flat_coef = flat_coef

    def forward(self,
                featuremaps_teacher: List[torch.Tensor],
                featuremaps_student: List[torch.Tensor],
                logits_teacher: torch.Tensor,
                logits_student: torch.Tensor) -> torch.Tensor:
        width_losses = []
        heght_losses = []
        for featuremap_teacher, featuremap_student in zip(featuremaps_teacher, featuremaps_student):
            width_losses.append(self.spacial_width_loss(featuremap_teacher, featuremap_student))
            heght_losses.append(self.spacial_height_loss(featuremap_teacher, featuremap_student))

        height_loss = torch.sum(torch.tensor(heght_losses))
        width_loss = torch.sum(torch.tensor(width_losses))

        flat_loss = self.pod_flat_loss(logits_teacher, logits_student)

        return self.width_coef * width_loss + self.height_coef*height_loss + self.flat_coef*flat_loss



