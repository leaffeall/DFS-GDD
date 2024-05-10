import argparse

from detectors import DETECTOR
from torch.utils.data import DataLoader
from dataset.datasets_train import *
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score
from tqdm import tqdm


def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    CM = confusion_matrix(label, prediction >= 0.5)
    acc = accuracy_score(label, prediction >= 0.5)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    return auc, TPR, FPR, acc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    interattributes = opt.inter_attribute.split('-')
    singleattributes = opt.single_attribute.split('-')

    test_data_names = ['DF', 'F2F', 'FST', 'FS','NT']
    test_data_paths = [
        './Deepfakes.csv',
        './Face2Face.csv',
        './FaceShifter.csv',
        './FaceSwap.csv',
        './NeuralTextures.csv'
    ]

    for i in range(1):
        model = model_class()
        if cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.to(device)

        checkpoints_path = f'./DFS.pth'
        ckpt = torch.load(checkpoints_path)
        model.load_state_dict(ckpt, strict=True)
        print(f'Loading model from: {checkpoints_path}')

        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        for test_data_name, test_data_path in zip(test_data_names, test_data_paths):
            test_dataset = ImageDataset_Test(
                test_data_path, data_transforms['test'], test_set='ff++')

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

            auc, TPR, FPR, acc = classification_metrics(
                label_list, pred_probs_list)

            print('Acc: {:.4f}, AUC: {:.4f}, TPR: {:.4f}, FPR: {:.4f}'.format(
                acc, auc, TPR, FPR))

            # Save results to file
            with open('./result.txt', 'a') as f:
                f.write(f"Model: {i}, Dataset: {test_data_name}, Acc: {acc}, AUC: {auc}, FPR: {FPR}\n")