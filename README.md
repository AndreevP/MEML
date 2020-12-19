# MEML
Matrix Exponential for Machine Learning

by Pavel Andreev, Peter Mokrov, Alexander Kagan, Nadezhda Alsahanova and Sofya Ivolgina

This is an accompanying repository for Numerical Linear Algebra course project. The main goal of this project is to study the applicability of different methods of numercial evaluation of the matrix exponential to machine learning tasks. See project report for more details.

**Project Report**: [pdf](https://drive.google.com/file/d/1AoGfMXKVplaxxKazk3kX9us1z8ZypY7t/view?usp=sharing)

*Note: most of the code for experiments with Generative Flows was taken from the official [implementation](https://github.com/changyi7231/MEF), while code for Intelligent Matrix Exponentiation was rewritten in PyTorch (original paper provided only Tensorflow [implementation](https://github.com/google-research/google-research/tree/master/m_layer)).*

## Experiments launching

**Generative flows**

To launch experiment related to generative flows use `run_flow.py` script.
An example of script run is below:

```bash
> python run_flow.py --matrix_exp optimized_taylor --matmuls 5 --mode test --device cuda:1
```
One can find available arguments and options by running the script with `--help` flag. The pretrained weights can be downloaded by this [link](https://drive.google.com/file/d/190nJdKhR50mZNAYmV_GyQ2rQRu3YS_us/view?usp=sharing) .

**MLayer**

Our results, related to `MLayer` presented in the following notebooks:

* [spiral_MLayer](./MLayer/M_Layer_Spiral_Experiments.ipynb) : this notebook presents our experiments with `Swiss roll dataset`

* [periodic_MLayer](./MLayer/M_Layer_MinTemp_Experiments.ipynb) : this notebooks presents our experiments with `Daily Minimum Temperatures in Melbourne` periodic dataset
