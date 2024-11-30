# [Decoupling Forgery Semantics for Generalizable Deepfake Detection](https://arxiv.org/abs/2406.09739)

Wei Ye, Xinan He, and Feng Ding
_________________

This repository is the official implementation of our paper "Decoupling Forgery Semantics for Generalizable Deepfake Detection", which has been accepted by **BMVC 2024**🎉🎉. 

## 💡 Installation
You can run the following script to configure the necessary environment:

```
cd DFS-GDD
conda create -n DFSGDD python=3.9.0
conda activate DFSGDD
pip install -r requirements.txt
```

## 🎭 Dataset Preparation

We share the FF++, Celeb-DF, DFD, DFDC with demographic annotations from [paper](https://arxiv.org/pdf/2208.05845.pdf),  which be downloaded through this [link](https://purdue0-my.sharepoint.com/:f:/g/personal/lin1785_purdue_edu/EtMK0nfxMldAikDxesIo6ckBVHMME1iIV1id_ZsbM9hsqg?e=WayYoy). 

Or you can download these datasets from their official website and process them by following the below steps:
- Download [FF++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFD](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html) and [DFDC](https://ai.facebook.com/datasets/dfdc/) datasets
- Download annotations for these four datasets according to [paper](https://arxiv.org/pdf/2208.05845.pdf) and their [code](https://github.com/pterhoer/DeepFakeAnnotations), extract the demographics information of all images in each dataset. 
- Extract, align and crop face using [DLib](https://www.jmlr.org/papers/volume10/king09a/king09a.pdf), and save them to `/path/to/cropped_images/`
- Split cropped images in each dataset to train/val/test with a ratio of 60%/20%/20% without identity overlap.
- Generate faketrain.csv, realtrain.csv, fakeval.csv, realval.csv according to the following format:
  
		|- faketrain.csv
			|_ img_path,label,spe_label
				/path/to/cropped_images/imgxx.png, 1(fake), 1(Deepfakes)/2(Face2Face)/3(FaceSwap)/4(NeuralTextures)/5(FaceShifter)
				...

		|- realtrain.csv
			|_ img_path,label
				/path/to/cropped_images/imgxx.png, 0(real)
				...

		|- fakeval.csv
			|_ img_path,label,spe_label
				/path/to/cropped_images/imgxx.png, 1(fake), 1(Deepfakes)/2(Face2Face)/3(FaceSwap)/4(NeuralTextures)/5(FaceShifter)
				...

		|- realval.csv
			|_ img_path,label
				/path/to/cropped_images/imgxx.png, 0(real)
				...
		
- Generate test.csv according to following format:

		|- test.csv
			|- img_path,label
				/path/to/cropped_images/imgxx.png, 1(fake)/0(real)
				...

## 💫 Load Pretrained Weights
Before running the training code, please ensure that you have loaded the pre-trained weights. You can directly use the pre-trained models of [Xception](https://www.dropbox.com/scl/fi/mr3b2fksm2al1a8sjyf9t/xception-b5690688.pth?rlkey=6glri2bfj6djfdbmdf52g6ikq&st=om3vz7ru&dl=0) and [SwiftFormer_L1](https://www.dropbox.com/scl/fi/tezc62c3vdg44i6q79f2a/SwiftFormer_L1.pth?rlkey=1qkd695lbg4q18hxo3nsnzy6w&st=nstk1z2n&dl=0) provided by us, or you can download the *Xception* model pre-trained on ImageNet (through this [link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)) and select a pre-trained *SwiftFormer* model from this [repository](https://github.com/Amshaker/SwiftFormer). Alternatively, you can use your own pre-trained *Xception* and *SwiftFormer* models.

## 🔥 Train
To run the training code, you should first go to the [`./training/`](./training/) folder. The training process is divided into two stages. You should start by training the first stage by running [`train_stage1.py`](training/train_stage1.py):

```
cd training

python train_stage1.py 
```
After that, in the code for the second stage, you need to import the model obtained from the first stage training for further training:

```
cd training

python train_stage2.py 
```

You can adjust the parameters in [`train_stage1.py`](training/train_stage1.py) or [`train_stage2.py`](training/train_stage2.py) to specify the parameters, *e.g.,* training dataset, batchsize, learnig rate, *etc*.

`--lr`: learning rate, default is 0.0005. 

`--gpu`: gpu ids for training.

` --fake_datapath`: /path/to/faketrain.csv, fakeval.csv

` --real_datapath`: /path/to/realtrain.csv, realval.csv

`--batchsize`: batch size, default is 16.

`--dataname`: training dataset name: ff++.

`--model`: detector name: DFS.

## 🤩 Test
For model testing, we provide [`test_in.py`](training/test_in.py) for in-domain testing and [`test_cross.py`](training/test_cross.py) for cross-domain testing to evaluate our model.

`--test_path`: /path/to/test.csv 

`--test_data_name`: testing dataset name: ff++, celebdf, dfd, dfdc.

`--checkpoints`: /path/to/saved/model.pth 

`--savepath`: /where/to/save/results/ 

`--model_structure`: detector name: DFS.

`--batch_size`: testing batch size: default is 32.

## ❗Note
Please note that in files [`train_stage1.py`](training/train_stage1.py), [`train_stage2.py`](training/train_stage2.py), [`DFS_1.py`](training/detectors/DFS_1.py), [`DFS_2.py`](training/detectors/DFS_2.py), [`test_in.py`](training/test_in.py) and [`test_cross.py`](training/test_cross.py), you will need to replace the paths in the code with your own file paths.

## 📦 Provided Models
|                    | File name                                          |
|--------------------|----------------------------------------------------|
| Pre_Xception       | [xception-b5690688.pth](https://www.dropbox.com/scl/fi/mr3b2fksm2al1a8sjyf9t/xception-b5690688.pth?rlkey=6glri2bfj6djfdbmdf52g6ikq&st=om3vz7ru&dl=0) |
| Pre_SwiftFormer_L1 | [SwiftFormer_L1.pth](https://www.dropbox.com/scl/fi/tezc62c3vdg44i6q79f2a/SwiftFormer_L1.pth?rlkey=1qkd695lbg4q18hxo3nsnzy6w&st=nstk1z2n&dl=0) |
| Our_Stage1         | [DFS_1_10.pth](https://www.dropbox.com/scl/fi/rzl2h3ljyjjaptn24bz0r/DFS_1_10.pth?rlkey=2dnn285zlwbyrrimlgkox3scu&st=cz6u44hd&dl=0) |
| Our_Stage2         | [DFS_2_8.pth](https://www.dropbox.com/scl/fi/w9gai5wgvlowygdlqk2we/DFS_2_8.pth?rlkey=7iqfv1y9d30cfst2mdv2wqvym&st=2acf0c3e&dl=0)   | 

## 📖 Citation
Please kindly consider citing our papers in your publications. 
```BibTeX
@inproceedings{Wei2024dfs,
    title={Decoupling Forgery Semantics for Generalizable Deepfake Detection},
    author={Wei Ye and Xinan He and Feng Ding},
    booktitle={British Machine Vision Conference (BMVC)},
    year={2024},
}
```