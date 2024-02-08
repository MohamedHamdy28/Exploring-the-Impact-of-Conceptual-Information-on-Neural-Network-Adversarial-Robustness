
from template_model import MLP, inception_v3, End2EndModel
from torch import nn
import torch

# Independent & Sequential Model
def ModelXtoC(pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, three_class):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                        three_class=three_class)

# Independent Model
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)

# Joint Model
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
                 use_relu, use_sigmoid):
    model1 = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                          n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                          three_class=(n_class_attr == 3))
    if n_class_attr == 3:
        model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)

# Standard Model
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux)

# Multitask Model
def ModelXtoCY(pretrained, freeze, num_classes, use_aux, n_attributes, three_class, connect_CY):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=False, three_class=three_class,
                        connect_CY=connect_CY)


class Sequential(nn.Module):
    def __init__(self, n_concepts, num_classes):
        super(Sequential, self).__init__()
        self.n_concepts = n_concepts
        self.g_model = ModelXtoC(pretrained=True, freeze=True, num_classes=num_classes, use_aux=False, n_attributes=n_concepts*2,
                                 expand_dim=1, three_class=False)
        self.f_model = ModelOracleCtoY(n_class_attr=1, n_attributes=n_concepts, num_classes=num_classes, expand_dim=128)

    def forward(self, x):
        x = self.g_model(x)
        print(x)
        # concatenate the output of the g_model which is a list of n_concepts
        x = torch.cat(x, dim=1)
        x = self.f_model(x)
        return x
    def load_g(self, path):
        self.g_model.load_state_dict(torch.load(path))
    
class Joint(nn.Module):
    def __init__(self, n_concepts, num_classes):
        super(Joint, self).__init__()
        self.n_concepts = n_concepts
        self.model = ModelXtoCtoY(n_class_attr=1, pretrained=True, freeze=True, num_classes=num_classes, use_aux=False,
                                  n_attributes=n_concepts, expand_dim=128, use_relu=False, use_sigmoid=False)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Standard(nn.Module):
    def __init__(self, num_classes):
        super(Standard, self).__init__()
        self.model = ModelXtoY(pretrained=True, freeze=True, num_classes=num_classes, use_aux=False)

    def forward(self, x):
        x = self.model(x)
        return x