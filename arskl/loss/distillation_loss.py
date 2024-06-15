import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops import reduce
from pyskl.models.cnns.resnet3d_slowfast import ResNet3dSlowFast
from pyskl.models.cnns.resnet3d_slowonly import ResNet3dSlowOnly
from pyskl.models.heads.simple_head import I3DHead

from ..model.builder import build_model
from ..registry import MODEL


class ResNet3dSO(nn.Module):
    def __init__(self, backbone_cfg, cls_head_cfg):
        super().__init__()
        self.backbone = ResNet3dSlowOnly(**backbone_cfg)
        self.cls_head = I3DHead(**cls_head_cfg)

    def forward(self, x):
        out = self.backbone(x)
        out = self.cls_head(out)
        return out


class ResNet3dSO_1(nn.Module):
    def __init__(self, backbone_cfg, cls_head_cfg):
        super().__init__()
        self.backbone = ResNet3dSlowFast(**backbone_cfg)
        self.cls_head = I3DHead(**cls_head_cfg)

    def forward(self, x):
        out = self.backbone(x)
        out = self.cls_head(out)
        return out


@MODEL.register_module()
class ResNet3d(nn.Module):
    def __init__(self, backbone_cfg, cls_head_cfg, ckpt_path):
        super().__init__()
        self.model = ResNet3dSO(backbone_cfg, cls_head_cfg)
        self.ckpt_path = ckpt_path

    def init_weights(self):
        if self.ckpt_path is not None:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt)

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        out = self.model(x)
        out = reduce(out, '(b s) c->b c', 'mean', s=s)
        return out


@MODEL.register_module()
class ResNet3d_1(nn.Module):
    def __init__(self, backbone_cfg, cls_head_cfg, ckpt_path):
        super().__init__()
        self.model = ResNet3dSO_1(backbone_cfg, cls_head_cfg)
        self.ckpt_path = ckpt_path

    def init_weights(self):
        if self.ckpt_path is not None:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt)

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        out = self.model(x)
        out = reduce(out, '(b s) c->b c', 'mean', s=s)
        return out


class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion, distillation_type, alpha, tau, teacher_cfg):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = build_model(teacher_cfg)
        self.teacher_model.init_weights()
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        loss = base_loss + distillation_loss * self.alpha
        return loss


class DiscriminationLoss(torch.nn.Module):
    def __init__(self, base_criterion, alpha, teacher_cfg):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = build_model(teacher_cfg)
        self.teacher_model.init_weights()
        self.alpha = alpha

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        base_loss = self.base_criterion(outputs, labels)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        teacher_loss = self.base_criterion(teacher_outputs, labels)
        loss = base_loss + teacher_loss * self.alpha
        return loss


class DiscriminationLoss2(torch.nn.Module):
    def __init__(self, base_criterion, distillation_type, alpha, tau, teacher_cfg):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = build_model(teacher_cfg)
        self.teacher_model.init_weights()
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = F.one_hot(labels, num_classes=60)
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        loss = base_loss + distillation_loss * self.alpha
        return loss


class DiscriminationLoss3(torch.nn.Module):
    def __init__(self, base_criterion, distillation_type, alpha, tau, teacher_cfg, num_classes):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = build_model(teacher_cfg)
        self.teacher_model.init_weights()
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.num_classes = num_classes

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = F.one_hot(labels, num_classes=self.num_classes)
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        pt = torch.exp(-base_loss)  # base_loss越大，pt越接近0，(1 - pt)越接近1
        if (1 - pt) > 0.35:  # 0.35(ntu60-Xsub)  0.25(ntu60-XView)
            focal_loss = torch.exp(1 - pt) * base_loss
        else:
            focal_loss = (1 - pt) ** 2 * base_loss
        loss = focal_loss + distillation_loss * self.alpha
        return loss


class DiscriminationLoss4(torch.nn.Module):
    def __init__(self, base_criterion, distillation_type, alpha, tau, teacher_cfg, num_classes):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = build_model(teacher_cfg)
        self.teacher_model.init_weights()
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.num_classes = num_classes

    def forward(self, inputs, outputs, labels):
        target = F.one_hot(labels, num_classes=self.num_classes)  # 真实标签
        with torch.no_grad():
            base_result = self.teacher_model(inputs)  # 基准路径输出
        relearn_result = outputs  # 再学习路径输出
        relearn_loss = self.base_criterion(relearn_result, labels)
        base_loss = self.base_criterion(base_result, labels)
        T = self.tau
        distillation_loss = F.kl_div(
            F.log_softmax(relearn_result / T, dim=1),
            F.log_softmax(base_result / T, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T) / relearn_result.numel()
        distillation_loss_hard = F.cross_entropy(relearn_result, base_result.argmax(dim=1))
        distillation_loss = distillation_loss * 0.9 + distillation_loss_hard * 0.1
        pt = torch.exp(-base_loss)
        base_loss = (1 - pt) ** 2 * base_loss
        p = [1, 1.1, 9.9]
        loss = relearn_loss * p[0] + base_loss * p[1] + distillation_loss * p[2]  # 1.权值调整 2.蒸馏损失软硬结合
        # loss = relearn_loss + base_loss * (relearn_loss.detach() / base_loss.detach()) + distillation_loss * (
        #         relearn_loss.detach() / distillation_loss.detach())  # 1.权值调整 2.蒸馏损失软硬结合
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
