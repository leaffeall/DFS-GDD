import sys
from detectors import DETECTOR
import torch
from torch.optim import lr_scheduler
import numpy as np
import os.path as osp
from utils1 import Logger
from utils.bypass_bn import enable_running_stats, disable_running_stats
import torch.backends.cudnn as cudnn
from dataset.pair_dataset import pairDataset
import csv
import argparse
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score
from tqdm import tqdm
import torch.optim as optim


parser = argparse.ArgumentParser("Center Loss Example")
parser.add_argument('--weight', type=float, default=0.5,
                    help="alpha in dag-fdd, (0.0~1.0)")
parser.add_argument('--lr', type=float, default=0.0005,
                    help="learning rate for training")
parser.add_argument('--batchsize', type=int, default=16, help="batch size")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataname', type=str, default='ff++',
                    help='ff++, celebdf, dfd, dfdc')
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--fake_datapath', type=str,
                    default='./fake.csv')
parser.add_argument('--real_datapath', type=str,
                    default='./real.csv')
parser.add_argument("--continue_train", action='store_true')
parser.add_argument("--checkpoints", type=str, default='',
                    help="continue train model path")
parser.add_argument("--model", type=str, default='DFS',
                    help="model structure[resnet, xception, efficient, dsp-fwa]")

args = parser.parse_args()


###### different data transform for different backbone #######
if args.model == 'xception':
    from transform import xception_default_data_transforms as data_transforms
if args.model == 'efficient':
    from transform import xception_default_data_transforms as data_transforms
if args.model == 'resnet':
    from transform import resnet_default_data_transforms as data_transforms
if args.model == 'dspfwa':
    from transform import resnet_default_data_transforms as data_transforms
if args.model == 'DFS':
    from transform import DFS_default_data_transforms as data_transforms



face_dataset = {x: pairDataset(args.fake_datapath+'fake'+'{}.csv'.format(
    x), args.real_datapath+'real'+'{}.csv'.format(
    x), data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(
    dataset=face_dataset[x], batch_size=args.batchsize, shuffle=True, num_workers=8, collate_fn=face_dataset[x].collate_fn) for x in ['train', 'val']}
dataset_sizes = {x: len(face_dataset[x]) for x in ['train', 'val']}
device = torch.device('cuda:0')

# prepare the model (detector)
model_class = DETECTOR['DFS_1']


def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    acc = accuracy_score(label, prediction >= 0.5)
    return auc, acc


# train and evaluation
def train(model,  optimizer, scheduler, num_epochs, start_epoch):
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()

        total_loss = 0.0

        for idx, data_dict in enumerate(tqdm(dataloaders[phase],file=sys.stdout)):
            imgs, labels = data_dict['image'], data_dict['label']
            if 'label_spe' in data_dict:
                label_spe = data_dict['label_spe']
                data_dict['label_spe'] = label_spe.to(device)
            data_dict['image'], data_dict['label'] = imgs.to(
                device), labels.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                optimizer.zero_grad()
                preds = model(data_dict)
                losses = model.get_losses(data_dict, preds)
                losses = losses['overall']
                losses.backward()
                optimizer.step()
                model.get_high.reset_center()
                model.encoder_hf.get_high1.reset_center()
                model.encoder_hf.get_high2.reset_center()

            if idx % 50 == 0:
                # compute training metric for each batch data
                batch_metrics = model.get_train_metrics(data_dict, preds)
                print('#{} batch_metric{}'.format(idx, batch_metrics))

            total_loss += losses.item() * imgs.size(0)

        epoch_loss = total_loss / dataset_sizes[phase]
        print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

        # update learning rate
        if phase == 'train':
            scheduler.step()

        # evaluation
        # if (epoch+1) % 5 == 0:
        if (epoch+1) % 1 == 0:
            savepath = './checkpoints_gauss_DFS/'+args.model+'/' + \
                    args.dataname+'_'+'/lr'+str(args.lr)



            print()
            print('-' * 10)

            phase = 'val'
            model.eval()
            running_corrects = 0
            total = 0

            pred_label_list = []
            pred_probs_list = []
            label_list = []

            for idx, data_dict in enumerate(dataloaders[phase]):
                imgs, labels = data_dict['image'], data_dict['label']
                # FIXME: do not consider the specific label when testing
                # fix the label to 0 and 1 only
                labels = torch.where(data_dict['label'] != 0, 1, 0)
                if 'label_spe' in data_dict:
                    data_dict.pop('label_spe')

                data_dict['image'], data_dict['label'] = imgs.to(
                    device), labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(data_dict, inference=True)
                    _, preds_label = torch.max(preds['cls'], 1)
                    pred_probs = torch.softmax(
                        preds['cls'], dim=1)[:, 1]
                    total += data_dict['label'].size(0)
                    running_corrects += (preds_label ==
                                         data_dict['label']).sum().item()

                    preds_label = preds_label.cpu().data.numpy().tolist()
                    pred_probs = pred_probs.cpu().data.numpy().tolist()
                # losses = model.get_losses(data_dict, preds)
                pred_label_list += preds_label
                pred_probs_list += pred_probs
                label_list += labels.cpu().data.numpy().tolist()
                if idx % 50 == 0:
                    batch_metrics = model.get_test_metrics()
                    print('#{} batch_metric{{"acc": {}, "auc": {}, "eer": {}, "ap": {}}}'.format(idx,
                                                                                                 batch_metrics['acc'],
                                                                                                 batch_metrics['auc'],
                                                                                                 batch_metrics['eer'],
                                                                                                 batch_metrics['ap']))


            pred_probs_list = np.array(pred_probs_list)
            label_list = np.array(label_list)

            epoch_acc = running_corrects / total

            auc, _ = classification_metrics(
                label_list, pred_probs_list)

            print('Epoch {} Acc: {:.4f}  auc: {}'.format(
                epoch, epoch_acc,  auc))
            with open(savepath+"/val_metrics.csv", 'a', newline='') as csvfile:
                columnname = ['epoch', 'epoch_acc', 'AUC all']
                writer = csv.DictWriter(csvfile, fieldnames=columnname)
                writer.writerow({'epoch': str(epoch), 'epoch_acc': str(epoch_acc),  'AUC all': str(auc)})
            temp_model = savepath + "/" + args.model+str(epoch) + '.pth'
            torch.save(model.state_dict(), temp_model)


            print()
            print('-' * 10)

    return model, epoch


def main():

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    sys.stdout = Logger(osp.join('./checkpoints_gauss_DFS/'+args.model+'/' + \
                    args.dataname+'_'+'/lr'+str(args.lr)+'/log_training.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model = model_class()
    model.to(device)


    start_epoch = 0

    if args.continue_train and args.checkpoints != '':
        state_dict = torch.load(args.checkpoints)
        model.load_state_dict(state_dict)
        print('continue train from: ', args.checkpoints)
        start_epoch = int(
            ((args.checkpoints).split('/')[-1]).split('.')[0][8:])+1

    # optimize
    params_to_update = model.parameters()
    optimizer4nn = optim.SGD(params_to_update, lr=args.lr,
                             momentum=0.9, weight_decay=5e-03)

    optimizer = optimizer4nn
    print(params_to_update, optimizer)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.5)


    model, epoch = train(model, optimizer,
                         exp_lr_scheduler, num_epochs=20, start_epoch=start_epoch)


    if epoch == 19:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()
