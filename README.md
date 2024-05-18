# BNN Enhanced Few-Shot Fine-Tuning on Diffusion Models

This repository contains code for implementing Bayesian Neural Networks (BNNs) alongside several few-shot fine-tuning methods on diffusion models, including DreamBooth, LoRA, and OFT. Additionally, a demo class is provided to showcase the performance differences when using BNNs versus not using them.

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
