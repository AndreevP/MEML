# MEML
Matrix Exponential for Machine Learning

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