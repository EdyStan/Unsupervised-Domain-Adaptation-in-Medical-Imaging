import torch
import torch.nn as nn


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None 

class DomainClassifier(nn.Module):
    def __init__(self, in_features):
        super(DomainClassifier, self).__init__()
        self.domain_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.domain_fc(x)
    

class UDARetinaNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.retinanet = model
        self.domain_classifier_p = DomainClassifier(49152*4)
        self.domain_classifier_loss = nn.BCELoss()
        self.grl = GradientReversalLayer.apply

    def forward(self, images, targets=None, domain='source', alpha=1.0):
        if self.training:
            # detection loss
            loss_dict = self.retinanet(images, targets)
            det_loss = sum(loss for loss in loss_dict.values())

            # extract backbone features
            images_transformed, _ = self.retinanet.transform(images, targets)
            feats = self.retinanet.backbone(images_transformed.tensors)

            # pool & flatten
            p3 = nn.AdaptiveAvgPool2d((16, 16))(feats['0']).flatten(1)
            p4 = nn.AdaptiveAvgPool2d((16, 16))(feats['1']).flatten(1)
            p5 = nn.AdaptiveAvgPool2d((16, 16))(feats['2']).flatten(1)
            p   = torch.cat([p3, p4, p5], dim=1)

            # GRL + domain classifier
            p_rev = self.grl(p, alpha)
            logits = self.domain_classifier_p(p_rev)             
            probs  = torch.sigmoid(logits)                       

            # BCE target labels (0 for source, 1 for target)
            batch_size = p3.size(0)
            domain_val  = 0.0 if domain == 'source' else 1.0
            labels      = torch.full((batch_size, 1), domain_val, device=probs.device)

            # compute BCE loss on probs
            domain_loss = 0.3 * self.domain_classifier_loss(probs, labels)

            return det_loss, domain_loss

        else:
            return self.retinanet(images)