# Unsupervised Meta-Domain Adaptation for Fashion Retrieval (UDMA)

### UDMA-Evaluation

Feature extraction using pre-trained Model (UDMA-MLP)

    CUDA_VISIBLE_DEVICES=0 python test_mlp.py  --WS=WS5 --model-name=DeepFashion --comb=L12 --optimizer=ADAM --eval-dataset=Street2Shop --load-epoch=45000 --batch-size=2000 --resume --finch-part=0

### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{udma,
        author    = {Authors}, 
        title     = {Unsupervised Meta-Domain Adaptation for Fashion Retrieval}, 
        booktitle = {Preprint},
        year      = {2020}
    }
