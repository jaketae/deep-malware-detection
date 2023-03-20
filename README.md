# Neural Network Malware Binary Classification

PyTorch implementation of [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435), [Learning the PE Header, Malware Detection with Minimal Domain Knowledge](https://arxiv.org/abs/1709.01471), and other derived models for malware detection.

All model checkpoints are available at [`assets/checkpoints`](assets/checkpoints).

## Quickstart

1. Clone this repository via

```
$ git clone https://github.com/jaketae/deep-malware-detection.git
$ cd pytorch-malware-detection
```

2. Create a Python virtual environment and install dependencies.

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -U pip wheel # update pip
$ pip install -r requirements.txt
```

3. Prepare PE files. `src/bin` provides scrapers to download malware. For instance, to download files from [dalswerk](https://das-malwerk.herokuapp.com), run

```
$ python -m src.bin.dasmalwerk
```

By default, this will download the files under the `raw` folder of the root directory.

4. Train the model.

```
$ cd src/deep_malware_detection
$ python train.py --benign_dir=YOUR_PATH_TO_BENIGN --malware_dir=YOUR_PATH_TO_MALWARE
```

## Data

This project was developed in late 2020, and unfortunately I lost access to the server where I collected data and ran experiments. While replicating all training data exactly may be infeasible, here are some resources for data collection.

1. [Wikidll.com](https://wikidll.com): Online website with downloadable benign `.dll` files. [Scraper](src/bin/dll.py).
2. [Dasmalwerk](https://das-malwerk.herokuapp.com): Online website with downloadable malware for research. [Scraper](src/bin/dasmalwerk.py).
3. [Malshare.com](https://malshare.com): Online website with downloadable malware for research. [Scraper](src/bin/malshare.py).
4. [EMBER](https://github.com/elastic/ember): Open dataset for malware detection research.
5. [Kaggle dataset](https://www.kaggle.com/datasets/amauricio/pe-files-malwares): PE file dataset availalbe on Kaggle, including both benign and malicious files.


## Implementation Notes

1. While Raff et. al used LSTMs for the sequential model, we tested both GRU and LSTMs and found that the former was easier to train.
2. We combined models presented in the two papers to derive a custom model that uses concatenated feature vector produced by the entry point 1D-CNN layer as well as the RNN units that follow. We denote these custom models with a "Res" prefix in the table below.
3. We also further develop the attention-based model in Raff et. al with this residual approach.
4. Due to computational constraints, we decided to only use PE file headers up to their 4096th bytes, thus creating a 4096 dimensional sequential feature vector for every file.

## Results

Presented below is a table detailing the performance of each model.

| Architecture   | Acc | F1   |
| -------------- | --- | ---- |
| MalConvBase    | 91  | .931 |
| MalConv+       | 94  | .951 |
| MalConv+ (E16) | 93  | .944 |
| MalConv+ (W64) | 94  | .949 |
| MC+ (E16,W64)  | 94  | .950 |
| MC+ (C256)     | 91  | .930 |
| GRU-CNN        | 93  | .946 |
| BiGRU-CNN      | 91  | .931 |
| GRU-CNN (H128) | 93  | .946 |
| ResGRU-CNN     | 94  | .948 |
| AttnGRU-CNN    | 94  | .952 |
| AttnResGRU-CNN | 94  | .952 |

For visualizations of training and model evaluation, refer to images in the `figures` directory.

## Contributing

The coding style is dictated by [black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/). You can apply them via 

```
# pip install black isort
make style
```

Please feel free to submit issues or pull requests.

## Citation

If you find this repository helpful for your research, please cite as follows.

```
@misc{dmd,
	title        = {Deep Malware Detection: A neural approach to malware detection in portable executables},
	author       = {Tae, Jaesung},
	year         = 2020,
	howpublished = {\url{https://github.com/jaketae/deep-malware-detection}}
}
```

## References

```
@misc{raff2017malware,
	title        = {Malware Detection by Eating a Whole EXE},
	author       = {Edward Raff and Jon Barker and Jared Sylvester and Robert Brandon and Bryan Catanzaro and Charles Nicholas},
	year         = 2017,
	eprint       = {1710.09435},
	archiveprefix = {arXiv},
	primaryclass = {stat.ML}
}
@article{Raff_2017,
	title        = {Learning the PE Header, Malware Detection with Minimal Domain Knowledge},
	author       = {Raff, Edward and Sylvester, Jared and Nicholas, Charles},
	year         = 2017,
	journal      = {Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security - AISec  ’17},
	publisher    = {ACM Press},
	doi          = {10.1145/3128572.3140442},
	isbn         = 9781450352024,
	url          = {http://dx.doi.org/10.1145/3128572.3140442}
}
```

## License

Released under the [MIT License](LICENSE).
