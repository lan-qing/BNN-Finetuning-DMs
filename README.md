# Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks (KDD 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2405.19931-b31b1b.svg)](https://arxiv.org/abs/2405.19931)

This repository provides an implementation of **Bayesian Neural Network (BNN)–based fine-tuning for diffusion models**, as proposed in our paper. It is designed to reproduce the key empirical results on mitigating the *corruption stage* observed during few-shot fine-tuning, by applying variational Bayesian training on top of existing personalization methods such as DreamBooth, LoRA, and OFT. The implementation follows the paper setup and introduces no additional inference-time cost.


## Requirements

To install the required libraries, execute the following command:

```
pip install -r requirements.txt
```




## Usage

### Baseline Models
To fine-tune a model using the DreamBooth, LoRA, or OFT methods without BNNs, run the appropriate script:

```
bash run_scripts_baseline_dreambooth/lora/oft.sh.

```


### Models with BNNs
To fine-tune a model using the DreamBooth, LoRA, or OFT methods with BNNs, execute the following script:


```
bash run_scripts_bayes_dreambooth/lora/oft.sh.

```



## Results
The generated images can be found in the `logs` directory. Results are evaluated using the following metrics: Clip-I, Clip-T, Dino, Lpips, and Clip-IQA.

## License
This project is licensed under the Apache License 2.0.

## Reference
If you find this repository or our work helpful, please consider citing:

```bibtex
@article{wu2024exploring,
  title={Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks},
  author={Wu, Xiaoyu and Zhang, Jiaru and Hua, Yang and Lyu, Bohan and Wang, Hao and Song, Tao and Guan, Haibing},
  journal={arXiv preprint arXiv:2405.19931},
  year={2024}
}

