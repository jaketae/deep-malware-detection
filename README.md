# Neural Network Malware Binary Classification

PyTorch implementation of [1] [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435), [2] [Learning the PE Header, Malware Detection with Minimal Domain Knowledge](https://arxiv.org/abs/1709.01471), and other derived custom models for malware detection.

## Quickstart

Clone this repository via

```
git clone https://github.com/jaketae/deep-malware-detection.git
cd pytorch-malware-detection
```

Then, a Python virtual environment:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you have [pipenv](https://pypi.org/project/pipenv/), you can also type

```
pipenv install -r requirements.txt
```

Train the model in Jupyter notebook titled `run.ipynb`, or start training through the terminal via

```
python train.py
```

Optional flags are documented below.

## Implementation Notes

1. While [2] used LSTMs for the sequential model, we tested both GRU and LSTMs and found that the former was easier to train.
2. We combined models presented in papers [1] and [2] to derive a custom model that uses concatenated feature vector produced by the entry point 1D-CNN layer as well as the RNN units that follow. We denote these custom models with a "Res" prefix in the table below.
3. We also further develop the attention-based model in [2] with this residual approach.
4. While the [1] used the entire binary of PE files, our approach more closely resembles that of [2]. Due to computational constraints, we decided to only use PE file headers up to their 4096th bytes, thus creating a 4096 dimensional sequential feature vector for every file.

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

The coding style is dictated by [black](https://black.readthedocs.io/en/stable/). Depending on development environment, you can toggle format-on-save options in your code editor or set up [pre-commit hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) to make the linter run on every push.

Please feel free to submit issues or pull requests if you find bugs or ways to optimize the code base. Emails to jaesungtae@gmail.com is also welcome!

## References

[1] Malware Detection by Eating a Whole EXE

```
@misc{raff2017malware,
      title={Malware Detection by Eating a Whole EXE},
      author={Edward Raff and Jon Barker and Jared Sylvester and Robert Brandon and Bryan Catanzaro and Charles Nicholas},
      year={2017},
      eprint={1710.09435},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

[2] Learning the PE Header, Malware Detection with Minimal Domain Knowledge

```
@article{Raff_2017,
   title={Learning the PE Header, Malware Detection with Minimal Domain Knowledge},
   ISBN={9781450352024},
   url={http://dx.doi.org/10.1145/3128572.3140442},
   DOI={10.1145/3128572.3140442},
   journal={Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security - AISec  ’17},
   publisher={ACM Press},
   author={Raff, Edward and Sylvester, Jared and Nicholas, Charles},
   year={2017}
}
```
