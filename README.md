# Unsupervised Meta-Domain Adaptation for Fashion Retrieval (UDMA)

### UDMA-MLP: Evaluation

#### Feature extraction using pre-trained Model (UDMA-MLP)
    >> CUDA_VISIBLE_DEVICES=0 python test_mlp.py  --WS=WS5 --model-name=DeepFashion --comb=L12 --optimizer=ADAM --eval-dataset=Street2Shop --load-epoch=45000 --batch-size=2000 --resume --finch-part=0

#### Quantitative Results
    >> cd evaluation_scripts
    >> eval_df_retrieval('DeepFashion', 'DeepFashion_ADAM_ALL', 60, 'X', 'regular') % DF test set
    mAP = 0.3075, r1 precision = 0.3107,  r5 precision = 0.5209,  r10 precision = 0.5994, r20 precision = 0.6712,  r50 precision = 0.7603
    >> eval_final_s2s_retrieval('Street2Shop', 'DeepFashion_ADAM_ALL', 60, 'X', 'regular') % DF-BL
    mAP = 0.2283, r1 precision = 0.3298,  r5 precision = 0.4470,  r10 precision = 0.4883, r20 precision = 0.5355, r50 precision = 0.5921
    >> eval_final_mlp_s2s_retrieval('Street2Shop', 'DeepFashion_ADAM_ALL', 60 , 'X', 'regular', 'L12_0_WS5', 45000) % UDMA-MLP
    mAP = 0.2430, r1 precision = 0.3592,  r5 precision = 0.4761,  r10 precision = 0.5241, r20 precision = 0.5644, r50 precision = 0.6210
### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{udma,
        author    = {Authors}, 
        title     = {Unsupervised Meta-Domain Adaptation for Fashion Retrieval}, 
        booktitle = {Preprint},
        year      = {2020}
    }
