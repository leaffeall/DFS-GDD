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
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F


from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='DFS_2')
class DFS_2(AbstractDetector):
    def __init__(self):
        super().__init__()

        self.num_classes = 2
        self.encoder_feat_dim = 2048
        self.half_fingerprint_dim = self.encoder_feat_dim//2


        self.encoder_f = self.build_backbone()

        self.loss_func = self.build_loss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fusion2 = fusion()
        self.get_high=GaussianFilter(3, 1.0,in_channels=3)


        self.discriminator = nn.Sequential(
            nn.Conv2d(self.encoder_feat_dim * 2, 512, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()

        uni_task_number = 6

        # head
        self.head_uni = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=uni_task_number
        )
        self.head_sha = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=self.num_classes
        )


        self.block_uni = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )
        self.block_sha = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )





    def build_backbone(self):#high
        # prepare the backbone

        backbone_class = BACKBONE['xception']

        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})
        state_dict = torch.load(
            './xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)

        state_dict1 = torch.load(
            './xception-b5690688.pth')
        for name, weights in state_dict1.items():
            if 'pointwise' in name:
                state_dict1[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict1 = {k.split('.')[0] + 's.' + '.'.join(k.split('.')[1:]): v for k, v in state_dict1.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict1, False)
        logger.info('Load pretrained model successfully!')


        return backbone

    def build_backbone1(self):#low
        # prepare the backbone

        backbone_class = BACKBONE['SwiftFormer_L1']

        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})
        weights_part2 = torch.load("SwiftFormer_L1")
        model_dict = backbone.state_dict()
        pretrained_dict2 = {k: v for k, v in weights_part2.items() if k in model_dict}
        model_dict.update(pretrained_dict2)
        backbone.load_state_dict(model_dict)
        return backbone



    def build_backbone3(self):
        # prepare the backbone

        backbone_class = BACKBONE['SwiftFormer_fusion']

        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})

        return backbone



    def build_loss(self):

        cls_loss_class = LOSSFUNC['cross_entropy']
        uni_loss_class = LOSSFUNC['cross_entropy']
        con_loss_class = LOSSFUNC['contrastive_regularization']
        rec_loss_class = LOSSFUNC['l1loss']
        cls_loss_func = cls_loss_class()
        uni_loss_func = uni_loss_class()
        con_loss_func = con_loss_class(margin=3.0)
        rec_loss_func = rec_loss_class()
        loss_func = {
            'cls': cls_loss_func,
            'uni': uni_loss_func,
            'con': con_loss_func,
            'rec': rec_loss_func
        }
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        cat_data = data_dict['image']
        high = self.get_high(cat_data)
        f_all = self.encoder_f.features(cat_data,high)


        feat_dict = {'forgery': f_all}
        return feat_dict

    def classifier(self, features: torch.tensor) -> torch.tensor:
        # split the features into the unique and common forgery
        f_uni = self.block_uni(features)
        f_share = self.block_sha(features)
        return f_uni, f_share


    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if 'recontruction_imgs' in pred_dict:
            return self.get_train_losses(data_dict, pred_dict)
        else:  # test mode
            return self.get_test_losses(data_dict, pred_dict)


    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:

        real_img, fake_img = pred_dict['real_fea'], pred_dict['fake_fea']

        # get the reconstruction imgs
        reconstruction_image_1, \
            reconstruction_image_2, \
            self_reconstruction_image_1, \
            self_reconstruction_image_2 \
            = pred_dict['recontruction_imgs']
        # get label
        label = data_dict['label']
        label_uni = data_dict['label_uni']

        # get pred
        pred = pred_dict['cls']
        pred_uni = pred_dict['cls_uni']


        # 1. classification loss for common features
        loss_sha = self.loss_func['cls'](pred, label)

        # 2. classification loss for unique features
        loss_uni = self.loss_func['uni'](pred_uni, label_uni)

        # 3. reconstruction loss
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

        # 4. constrative loss
        common_features = pred_dict['feat']
        unique_features = pred_dict['feat_uni']
        loss_con = self.loss_func['con'](
            common_features, unique_features, label_uni)


        loss = loss_sha + 0.3 * loss_reconstruction + 0.1 * loss_uni + 0.05 * loss_con


        loss_dict = {
            'overall': loss,
            'common': loss_sha,
            'uni': loss_uni,
            'reconstruction': loss_reconstruction,
        }
        return loss_dict

    def get_test_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get label
        label = data_dict['label']
        # get pred
        pred = pred_dict['cls']
        # for test mode, only classification loss for common features
        loss = self.loss_func['cls'](pred, label)
        loss_dict = {'common': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy

        # get pred and label
        label = data_dict['label']
        pred = pred_dict['cls']
        label_uni = data_dict['label_uni']
        pred_uni = pred_dict['cls_uni']

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())
        acc_uni = get_accracy(label_uni.detach(), pred_uni.detach())
        metric_batch_dict = {'acc_com': acc, 'acc_uni': acc_uni,
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
        forgery_features = features[
            'forgery']
        f_uni, f_share = self.classifier(forgery_features)


        if inference:
            # inference only consider share loss
            out_sha, sha_feat = self.head_sha1(f_share)
            prob_sha = torch.softmax(out_sha, dim=1)[:, 1]
            self.prob.append(
                prob_sha
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
            _, prediction_class = torch.max(out_sha, 1)
            correct = (prediction_class == data_dict['label']).sum().item()

            self.correct += correct
            self.total += data_dict['label'].size(0)

            pred_dict = {'cls': out_sha, 'feat': sha_feat}
            return pred_dict

        f_all = features['forgery']

        real_fea, fake_fea = f_all.chunk(2, dim=0)

        c2, c1 = f_share.chunk(2, dim=0)
        c22, c11 = f_uni.chunk(2, dim=0)


        self_reconstruction_image_1 = self.fusion1(c1, c11)#f

        self_reconstruction_image_2 = self.fusion1(c2, c22)#r

        reconstruction_image_1 = self.fusion1(c2, c11)#r

        reconstruction_image_2 = self.fusion1(c1, c22)#f



        # head for unique and sha
        out_uni, uni_feat = self.head_uni(f_uni)
        out_sha, sha_feat = self.head_sha(f_share)#检测器



        # build the prediction dict for each output
        pred_dict = {
            'cls': out_sha,
            'feat': sha_feat,
            'cls_uni': out_uni,
            'feat_uni': uni_feat,
            'recontruction_imgs': (
                reconstruction_image_1,
                reconstruction_image_2,
                self_reconstruction_image_1,
                self_reconstruction_image_2
            ),
            'real_fea': (
                real_fea
            ),
            'fake_fea': (
                fake_fea
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

class fusion(nn.Module):
    def __init__(self, in_f=1024, hidden_dim=2048, out_f=2048):
        super(fusion, self).__init__()
        self.conv2dx = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True))
        self.conv2dy = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True))
        self.conv2dx1 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True))
        self.conv2dy1 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True))
        self.conv2dx2 = nn.Sequential(nn.Conv2d(hidden_dim, out_f, 1, 1))

    def forward(self, x,y):
        x = self.conv2dx(x)
        y = self.conv2dy(y)
        x = x + y
        x = self.conv2dx1(x)
        y = self.conv2dy1(y)
        x = x + y
        x = self.conv2dx2(x)
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


