# Dependency-Based Neural Reordering Model for Statistical Machine Translation
This repository contains codes to build a dependency-based neural reordering model for statistical machine translation.

If you use this code for your work, please cite this [paper](www.comp.nus.edu.sg/~chrhad/pub/aaai2017_neuraldep.pdf):
```
@inproceedings{hadiwinoto2017neuraldep,
	title = {A dependency-based neural reordering model for statistical machine translation},
	booktitle = {Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence},
	author = {Hadiwinoto, Christian and Ng, Hwee Tou},
	year = {2017},
	pages = {109--115},
}
```

REQUIREMENTS
------------
* python==2.7
* theano==1.0.3

Running the Sample Scripts
--------------------------

Downloading and extracting the pretrained embeddings (binary word2vec format):

```
bash download_emb.sh
```

Initializing the pretrained embeddings to be loaded to the model:

```
bash runfilterembedding.sh {linear,dep}
```
where `linear` refers to the standard `word2vec` embeddings and `dep` refers to the dependency-based `word2vec` embeddings [(Bansal et al., 2014)](https://www.aclweb.org/anthology/P14-2131), which we trained based on our best understanding.

The input data should consist of training and development (tuning) data. Each of these should contain one dependency-parsed source-language file in CoNLL-X format and one alignment file in Giza++ format, that is, `<src_pos>-<trg_pos> ... <src_pos>-<trg_pos>`. Please refer to the example shown for the training data, namely `train.zh.jdp` for the source-language file and `train.align` for the alignment file, and similarly for development data. We cannot release the training data due to the license.

The training instances should then be extracted from the training and development data (please replace with your real, reasonably-sized, data):

```
bash runprepare-swap.train.sh
bash runprepare-swap.dev.sh
```

Training is done by calling the Python program as shown in the following script:

```
bash runtrain.gen.sh <GPU1> <GPU2> <RANDOM_SEED>
```
