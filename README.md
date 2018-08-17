# HCSC
Cold-Start Aware User and Product Attention for Sentiment Classification

This TensorFlow code was used in the experiments of the research paper

**Reinald Kim Amplayo**, Jihyeok Kim, Sua Sung, and Seung-won Hwang. **Cold-Start Aware User and Product Attention for Sentiment Classification**. _ACL_, 2018.

You will need to download the original data here: https://drive.google.com/open?id=1PxAkmPLFMnfom46FMMXkHeqIxDbA16oy

Also, if you want to use the sparse data, the `data` folder contains the IDs of the data instances used. Please refer to the readme file inside the `data` folder.

You will also need GloVe pretrained word vectors which can be downloaded here: http://nlp.stanford.edu/data/glove.840B.300d.zip

To run the code, use the following command:

`python src/hcsc_main.py data_dir base_model train_type`

where:
- `data_dir` is the data folder (e.g. `data/imdb`).
- `base_model` can be `cnn`, `rnn`, or `hcwe`. To use the full HCSC model, use `hcwe`.
- `train_type` is the extension of the dataset filename if the sparse datasets are used (e.g. `_zero2` if Sparse20 is used). If the original dataset is used, leave this argument blank

To cite the paper/code, please use this BibTex

```
@inproceedings{amplayo2018cold,
	Author = {Reinald Kim Amplayo and Jihyeok Kim and Sua Sung and Seung-won Hwang},
	Booktitle = {ACL},
	Location = {Melbourne, Australia},
	Year = {2018},
	Title = {Cold-Start Aware User and Product Attention for Sentiment Classification},
}
```

If you have questions, send me an email: rktamplayo at yonsei dot ac dot kr
