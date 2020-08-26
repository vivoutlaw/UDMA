# Unsupervised Meta-Domain Adaptation for Fashion Retrieval (UDMA)

### Prerequisites
- Tested on Linux: Ubuntu 16.04.1, and CUDA-9.0

### Getting Started
```bash
mkdir UDMA_Codebase && cd UDMA_Codebase
```
- Clone this repo:
```bash
git clone https://github.com/almazan/deep-image-retrieval.git
cd deep-image-retrieval/dirtorch
mkdir data && cd data
```
- Download the pre-trained model: [Resnet101-TL-GeM model](https://drive.google.com/open?id=1vhm1GYvn8T3-1C4SPjPNJOuTU9UxKAG6) available from [Deep Image Retrieval](https://github.com/almazan/deep-image-retrieval) and save it in the `data` folder.
- setup vitual environment
```bash
cd ../..
python3 -m venv venv
source venv/bin/activate
```
- Install packages
```bash
pip install torch torchvision
pip install pytorch-metric-learning
pip install pandas
pip install h5py matplotlib
pip install opencv-python
```
- Clone the `UDMA` repo in `deep-image-retrieval`
```bash
>> PYTHONPATH=$PYTHONPATH:PATH-TO/deep-image-retrieval
>> git clone https://github.com/vivoutlaw/UDMA.git
>> cd UDMA
```

## Datasets
Download the pre-extracted features for both `DeepFashion` and `Street2Shop` datasets using the following script. The feature were extracted using our pre-trained model on DeepFashion. Trainset (`ClusteringFeats`) and Testset (`Street2Shop` | `DeepFashion`)
```bash
bash ./features/download_dataset.sh Street2Shop
bash ./features/download_dataset.sh DeepFashion
bash ./features/download_dataset.sh ClusteringFeats
```

## UDMA-MLP: Pre-trained model evaluation
- `Feature extraction using pre-trained UDMA-MLP model`: 
```bash
>> CUDA_VISIBLE_DEVICES=0 python test_mlp.py  --WS=WS5 --model-name=DeepFashion --comb=L12 --optimizer=ADAM --eval-dataset=Street2Shop --load-epoch=45000 --batch-size=2000 --resume --finch-part=0
```
- `Quantitative Results: DF-BL, UDMA-MLP`: 
```bash
>> cd evaluation_scripts
>> eval_final_s2s_retrieval('Street2Shop', 'DeepFashion_ADAM_ALL', 60, 'X', 'regular') % DF-BL
mAP = 0.2283, r1 precision = 0.3298,  r5 precision = 0.4470,  r10 precision = 0.4883, r20 precision = 0.5355, r50 precision = 0.5921
>> eval_final_mlp_s2s_retrieval('Street2Shop', 'DeepFashion_ADAM_ALL', 60 , 'X', 'regular', 'L12_0_WS5', 45000) % UDMA-MLP
mAP = 0.2430, r1 precision = 0.3592,  r5 precision = 0.4761,  r10 precision = 0.5241, r20 precision = 0.5644, r50 precision = 0.6210
```

## UDMA-MLP: Finetuning MLP with pretrained DF-BL model weights
- `Weighting strategy used for MLP training`:
```bash
>> python -W ignore weighting_strategy_part1.py --finch-part=0
>> python -W ignore weighting_strategy_part1.py --finch-part=0
>> python -W ignore weighting_strategy_part2.py --comb=L12 --optimizer=ADAM --finch-part=0 
```
- `Script for UDMA-MLP training`:
```bash
>> CUDA_VISIBLE_DEVICES=0 python -W ignore train_mlp.py  --WS=WS5 --dataset=DeepFashion --comb=L12 --optimizer=ADAM  --num-threads=8 --batch-size=128 --lr=1e-4 --resume-df --load-epoch-df=60 --epochs=45000 --finch-part=0 --batch-category-size=12 
```
## Finetuning Full [Resnet101-TL-GeM model](https://drive.google.com/open?id=1vhm1GYvn8T3-1C4SPjPNJOuTU9UxKAG6) on DeepFashion dataset
In our work, we utilized both `train` and `val` set of DeepFashion for model train, and tested on `test` set. 
- Change `train_test_type` from `trainval`  to `train` for training the model only on `train` set.
- `Quantitative Results: DF test set`: 
```bash
>> cd evaluation_scripts
>> eval_df_retrieval('DeepFashion', 'DeepFashion_ADAM_ALL', 60, 'X', 'regular') % DF test set
mAP = 0.3075, r1 precision = 0.3107,  r5 precision = 0.5209,  r10 precision = 0.5994, r20 precision = 0.6712,  r50 precision = 0.7603
```
- `Script for training the full-model`:
```bash
>> CUDA_VISIBLE_DEVICES=0,1,2,3 python main_train_df.py --dataset=DeepFashion --df-comb=ALL --optimizer=ADAM --num-threads=8 --batch-size=128 --lr=1e-4 --epochs=60 --checkpoint=../dirtorch/data/Resnet101-TL-GeM.pt
```
- `Pre-trained model is available here`: [DeepFashion Resnet101-TL-GeM model](xxx)
- After the model is trained, we utilize the last `fc` layer of this model for `UDMA-MLP`.
- `Script for feature extraction`:
```bash
>> 
```



## Citation
If you find the code and datasets useful in your research, please cite:
```    
@inproceedings{udma,
     author    = {Vivek Sharma, Gabriela Csurka, Naila Murray, Diane Larlus, M. Saquib Sarfraz, and Rainer Stiefelhagen}, 
     title     = {Unsupervised Meta-Domain Adaptation for Fashion Retrieval}, 
     booktitle = {Preprint},
     year      = {2020}
}

@inproceedings{finch,
    author    = {M. Saquib Sarfraz, Vivek Sharma and Rainer Stiefelhagen}, 
    title     = {Efficient Parameter-free Clustering Using First Neighbor Relations}, 
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {8934--8943}
    year  = {2019}
}
```
