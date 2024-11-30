import argparse
from detectors import DETECTOR
from torch.utils.data import DataLoader
from dataset.datasets_train import *
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score
from tqdm import tqdm


def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    acc = accuracy_score(label, prediction >= 0.5)
    return auc, acc


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int,
                        default=32, help="size of the batches")
    parser.add_argument("--test_data_name", type=str,
                        default='dfdc')  # ff++ celebdf dfd dfdc
    parser.add_argument("--savepath", type=str,
                        default='results')
    parser.add_argument("--model_structure", type=str, default='DFS',
                        help="efficient,ucf_daw")

    opt = parser.parse_args()

    cuda = torch.cuda.is_available()

    model_class = DETECTOR['DFS_2']

    from transform import DFS_default_data_transforms as data_transforms


    test_data_names = ['ff++', 'celebdf', 'dfd', 'dfdc']
    test_data_paths = [
        '/home/user/local/yw/training/testff++.csv',
        '/home/user/local/yw/training/testcelebdf.csv',
        '/home/user/local/yw/training/testdfd.csv',
        '/home/user/local/yw/training/testdfdc.csv'
    ]

    for i in range(0,1):
        model = model_class()
        if cuda:
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            model.to(device)

        checkpoints_path = f'/home/user/local/yw/DFS_2_8.pth'
        ckpt = torch.load(checkpoints_path)
        model.load_state_dict(ckpt, strict=True)
        print(f'Loading model from: {checkpoints_path}')

        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        for test_data_name, test_data_path in zip(test_data_names, test_data_paths):
            test_dataset = ImageDataset_Test(
                test_data_path, data_transforms['test'], test_data_name)

            test_dataloader = DataLoader(
                test_dataset, batch_size=opt.batch_size, shuffle=False)

            print("Testing on:", test_data_name)
            print('-' * 10)
            print('Total batches:', len(test_dataloader))

            running_corrects = 0
            total = 0

            pred_label_list = []
            pred_probs_list = []
            label_list = []
            model.eval()
            for data_dict in tqdm(test_dataloader,position=0):
                inputs, labels = data_dict['image'], data_dict["label"]
                data_dict['image'], data_dict["label"] = inputs.to(
                    device), labels.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(data_dict, inference=True)
                    _, preds_label = torch.max(preds['cls'], 1)
                    pred_probs = torch.softmax(
                        preds['cls'], dim=1)[:, 1]
                    total += inputs.size(0)
                    preds_label = preds_label.to(device)
                    running_corrects += (preds_label ==
                                         labels.data).sum().item()

                    preds_label = preds_label.cpu().data.numpy().tolist()
                    pred_probs = pred_probs.cpu().data.numpy().tolist()
                pred_label_list += preds_label
                pred_probs_list += pred_probs
                label_list += labels.cpu().data.numpy().tolist()

            pred_label_list = np.array(pred_label_list)
            pred_probs_list = np.array(pred_probs_list)
            label_list = np.array(label_list)
            epoch_acc = running_corrects / total

            auc, acc = classification_metrics(
                label_list, pred_probs_list)

            print('Acc: {:.4f}, AUC: {:.4f}'.format(
                acc, auc))

            # Save results to file
            with open('./result.txt', 'a') as f:
                f.write(f"Model: {i}, Dataset: {test_data_name}, Acc: {acc}, AUC: {auc}\n")