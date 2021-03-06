import torch
import torch.nn as nn
class CombineModel(nn.Module):
    def __init__(self, img_block, classifier_articleType, classifier_baseColour, freeze_conv=False):
        assert img_block and classifier_articleType and classifier_baseColour
        super().__init__()
        self.img_block = img_block
        if freeze_conv:
            for param in self.img_block.parameters():
                param.requires_grad = False
        self.classifier_articleType = classifier_articleType
        self.classifier_baseColour = classifier_baseColour
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, X, get_img_features=False):
        X_img = X
        X_img = self.img_block(X_img)
        X_img = self.avgpool(X_img).flatten(1)
        if get_img_features:
            return self.classifier_articleType(X_img), self.classifier_baseColour(X_img), X_img
        return self.classifier_articleType(X_img), self.classifier_baseColour(X_img)
    
# class TripletCombineModel(nn.Module):
#     def __init__(self, backbone_block, classifier, freeze_conv=False):
#         assert backbone_block and classifier
#         super().__init__()
#         self.backbone_block = backbone_block
#         self.classifier = classifier
#         if freeze_conv:
#             for param in self.backbone_block.parameters():
#                 param.requires_grad = False
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
    
#     def forward(self, X):
#         articleType, baseColour = self.backbone_block(X)
#         articleType = self.avgpool(articleType).flatten(1)
#         baseColour = self.avgpool(baseColour).flatten(1)
        
#         return self.classifier(articleType + baseColour)

class TripletCombineModel(nn.Module):
    def __init__(self, classifier, freeze_conv=False):
        assert classifier
        super().__init__()
        self.classifier = classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, X):
        return self.classifier(X)