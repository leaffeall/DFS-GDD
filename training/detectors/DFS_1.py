'''

# description: 

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation
'''
import logging
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='DFS_1')
class DFS_1(AbstractDetector):
    def __init__(self):
        super().__init__()

        self.num_classes = 2
        self.encoder_feat_dim = 2048
        self.half_fingerprint_dim = self.encoder_feat_dim//2

        self.encoder_content = self.build_backbone1()
        self.encoder_f = self.build_backbone()


        self.loss_func = self.build_loss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0


        self.fusion_stage1 = self.build_backbone3()
        self.get_high=GaussianFilter(3, 1.0,in_channels=3)


        self.sigmoid = nn.Sigmoid()

        # head
        self.head_all = Head(
            in_f=self.half_fingerprint_dim * 2,
            hidden_dim=self.encoder_feat_dim*2,
            out_f=self.num_classes
        )




    def build_backbone(self):
        # prepare the backbone

        backbone_class = BACKBONE['xception']

        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})
        state_dict = torch.load(
            '/home/user/local/yw/training/pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)

        state_dict1 = torch.load(
            '/home/user/local/yw/training/pretrained/xception-b5690688.pth')
        for name, weights in state_dict1.items():
            if 'pointwise' in name:
                state_dict1[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict1 = {k.split('.')[0] + 's.' + '.'.join(k.split('.')[1:]): v for k, v in state_dict1.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict1, False)
        logger.info('Load pretrained model successfully!')


        return backbone

    def build_backbone1(self):
        backbone_class = BACKBONE['SwiftFormer_L1']

        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})
        weights_part2 = torch.load("/home/user/local/yw/training/pretrained/SwiftFormer_L1.pth")
        model_dict = backbone.state_dict()
        pretrained_dict2 = {k: v for k, v in weights_part2.items() if k in model_dict}
        model_dict.update(pretrained_dict2)
        backbone.load_state_dict(model_dict)
        return backbone



    def build_backbone3(self):

        backbone_class = BACKBONE['SwiftFormer_fusion']

        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})

        return backbone



    def build_loss(self):

        cls_loss_class = LOSSFUNC['cross_entropy']
        rec_loss_class = LOSSFUNC['l1loss']
        cls_loss_func = cls_loss_class()
        rec_loss_func = rec_loss_class()

        loss_func = {
            'cls': cls_loss_func,
            'rec': rec_loss_func
        }
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        cat_data = data_dict['image']
        high = self.get_high(cat_data)
        c_all = self.encoder_content.forward(cat_data)
        f_all = self.encoder_f.features(cat_data,high)
        feat_dict = {'forgery': f_all, 'content': c_all}
        return feat_dict


    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if 'recontruction_imgs' in pred_dict:
            return self.get_train_losses(data_dict, pred_dict)
        else:  # test mode
            return self.get_test_losses(data_dict, pred_dict)



    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        cat_data = data_dict['image']

        real_img, fake_img = cat_data.chunk(2, dim=0)
        # get the reconstruction imgs
        reconstruction_image_1, \
            reconstruction_image_2, \
            self_reconstruction_image_1, \
            self_reconstruction_image_2 \
            = pred_dict['recontruction_imgs']
        # get label
        label = data_dict['label']

        # get pred
        pred = pred_dict['cls']


        # 1. classification loss for all forgery semantics
        loss_all = self.loss_func['cls'](pred, label)

        # 2. reconstruction loss
        self_loss_reconstruction_1 = self.loss_func['rec'](
            fake_img, self_reconstruction_image_1)
        self_loss_reconstruction_2 = self.loss_func['rec'](
            real_img, self_reconstruction_image_2)
        cross_loss_reconstruction_1 = self.loss_func['rec'](
            fake_img, reconstruction_image_2)
        cross_loss_reconstruction_2 = self.loss_func['rec'](
            real_img, reconstruction_image_1)
        loss_reconstruction = \
            self_loss_reconstruction_1 + self_loss_reconstruction_2 + \
            cross_loss_reconstruction_1 + cross_loss_reconstruction_2


        loss = loss_all +  0.3 * loss_reconstruction


        loss_dict = {
            'overall': loss,
            'all': loss_all,
            'reconstruction': loss_reconstruction
        }
        return loss_dict

    def get_test_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get label
        label = data_dict['label']
        # get pred
        pred = pred_dict['cls']
        # for test mode, only classification loss for all forgery semantics
        loss = self.loss_func['cls'](pred, label)
        loss_dict = {'all': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:

        label = data_dict['label']
        pred = pred_dict['cls']

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())

        metric_batch_dict = {'acc_all': acc,
                             'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct = 0
        self.total = 0
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict: dict, inference=False) -> dict:
        # split the features into the content and forgery
        features = self.features(data_dict)
        forgery_features, content_features = features[
            'forgery'], features['content']


        if inference:
            # inference only consider all_forgery loss
            out_all, all_feat = self.head_all(features['forgery'])
            prob_all = torch.softmax(out_all, dim=1)[:, 1]
            self.prob.append(
                prob_all
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(out_all, 1)
            correct = (prediction_class == data_dict['label']).sum().item()

            self.correct += correct
            self.total += data_dict['label'].size(0)

            pred_dict = {'cls': out_all, 'feat': all_feat}
            return pred_dict

        f_all = features['forgery']
        # reconstruction loss
        f2, f1 = f_all.chunk(2, dim=0)
        c2, c1 = content_features.chunk(2, dim=0)


        self_reconstruction_image_1 = self.fusion_stage1(c1, f1)

        self_reconstruction_image_2 = self.fusion_stage1(c2, f2)

        reconstruction_image_1 = self.fusion_stage1(c2, f1)

        reconstruction_image_2 = self.fusion_stage1(c1, f2)


        out_all, all_feat = self.head_all(f_all)

        prob_all = torch.softmax(out_all, dim=1)[:, 1]


        # build the prediction dict for each output
        pred_dict = {
            'cls': out_all,
            'prob': prob_all,
            'feat': all_feat,
            'recontruction_imgs': (
                reconstruction_image_1,
                reconstruction_image_2,
                self_reconstruction_image_1,
                self_reconstruction_image_2
            )
        }
        return pred_dict




class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
        return x


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size, sigma, in_channels=3):
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.in_channels = in_channels
        self.kernel = nn.Parameter(self.create_high_pass_kernel())

    def create_high_pass_kernel(self):
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        pad = self.kernel_size // 2
        for x in range(-pad, -pad + self.kernel_size):
            for y in range(-pad, -pad + self.kernel_size):
                kernel[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (self.sigma ** 2)))
        kernel /= (self.sigma ** 2 * np.pi * 2)
        kernel /= kernel.sum()
        identity_kernel = np.zeros((self.kernel_size, self.kernel_size))
        identity_kernel[pad, pad] = 1
        high_pass_kernel = identity_kernel - kernel
        high_pass_kernel /= -high_pass_kernel[pad, pad]
        return torch.FloatTensor(high_pass_kernel).unsqueeze(0).unsqueeze(0).repeat(self.in_channels, 1, 1, 1)

    def reset_center(self):
        with torch.no_grad():
            pad = self.kernel_size // 2
            center_values = self.kernel.data[:, :, pad, pad].clone()
            center_values.unsqueeze_(-1).unsqueeze_(-1)
            sum_except_center = torch.sum(self.kernel.data, dim=(2, 3), keepdim=True) - center_values
            self.kernel.data /= sum_except_center
            self.kernel.data[:, :, pad, pad] = -1


    def forward(self, x):
        device = x.device
        return F.conv2d(x, self.kernel.to(device), padding=self.kernel_size // 2, groups=self.in_channels)

