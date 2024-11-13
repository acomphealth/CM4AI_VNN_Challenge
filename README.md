# CM4AI VNN Challenge
1. Create a W&B account (https://wandb.ai)
2. Connect to compute environment
3. Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
4. Create a new conda environment
```bash
conda create -n vnn_challenge python=3.9
```
5. Activate environment and install dependencies
```bash
conda activate vnn_challenge
```
6. Clone the CM4AI_VNN_Challenge repository and install dependencies
```
git clone https://github.com/acomphealth/CM4AI_VNN_Challenge.git
cd /CM4AI_VNN_Challenge
pip install -r requirements.txt
```
7. Login to W&B with your API key
```bash
wandb login
```
8. Run template code and visualize results in W&B
```bash
python src/train.py
```

# Overview
## Problem Statement
You will be creating interpretable neural network models to predict cell line responses to palbociclib, a selective CDK4/6 inhibitor approved for use in patients with HER2-/HR+ advanced/metastatic breast cancer. These models are based on work published by Park et al. (2024), and will predict responses of 1,352 cancer cell lines from the somatic mutation and copy number amplification/deletion status across 718 cancer-related genes. The architecture of these neural networks is coupled to multiscale molecular pathways that are recurrently mutated in one or more cancer type. You will perform hyperparameter optimization and benchmark the performance of your model against the published palbociclib model. After training, you will query its architecture to identify molecular assemblies influencing palbociclib response, which you will then use in the LLM challenge to characterize genetic mechanisms. 

## Background
Alterations in the molecular machinery of healthy cells can disrupt the tightly regulated systems governing cellular growth. Dysregulation of these systems can cause cells to grow unchecked, a pathogenic phenotype commonly referred to as cancer. Anti-cancer agents aim to exploit the unique characteristics of cancer cells to selectively kill tumors and leave healthy tissues relatively unaffected. Recently, precision oncology drugs have been developed that selectively target and kill tumors harboring specific oncogenic mutations. However, other mutations in the drug target can also influence responses, as well as alterations in proteins that interact with the drug target either physically or functionally through the same or related molecular pathways. It is therefore necessary to build models that consider the states of many genes, but the complexity of these interactions makes predicting whether a tumor is likely to respond to a given drug exceedingly challenging. 

Past predictive accuracy, it is critical that these models are trustworthy. Models might make unexpected or counterintuitive predictions. Healthcare providers will not risk the lives of their patients without clear and plausible justification for why a prediction was made. Traditional neural networks are so-called “black boxes” because their architecture is too complex to fully grasp why a particular decision was made. Accordingly, visible neural networks (VNN) were proposed to make neural networks explicitly interpretable. Within a VNN, connections between neurons are limited based on an user-defined set of relationships relevant to the learning task. During training, models detect specific features in this architecture that are useful for predictions. After training, these features can be identified to understand whether the model gleaned relevant insights about the mechanisms influencing predictions. Such insights not only help validate VNN trustworthiness, but can also highlight novel associations.

## Drug response dataset
Consortia such as the Genomics of Drug Sensitivity in Cancer (GDSC) generate large-scale, publically available pharmacogenomics data useful for drug response prediction tasks. In addition to recording drug response metrics of primary cancer cell lines to cancer drugs, these data also provide extensive molecular profiles of cell line genomic features across many data modalities. Drug responses are typically reported as area under the drug response curve (AUDRC), which describes how a cell responds to a compound over several concentrations. The effects of a drug differ over varying concentrations (Fig Xa), but without analyzing responses over a range of doses it is impossible to know the extent to which a cell is affected by the drug (Fig Xb). To capture this information, the area under the dose response curved is summed across the tested concentration range (Fig Xc). GDSC screens all cell lines over a standardized dose range, then reports this value for each cell line. Models then evaluate the molecular profile of a cancer cell and predict this value.

## VNN architecture
The VNN architecture will embed systems biology knowledge from the Nested Systems in Tumors (NeST) protein-protein interaction network. This network was developed to identify recurrently mutated molecular pathways that occur in cancer cell lines, as opposed to networks for benign tissues. By using a cancer-specific network, the VNN will likely discover response mechanisms likely to occur in pathogenic cells. When limited to the 718 cancer-related genes in our gene set, the NeST hierarchy contains 162 molecular pathways over biological scales ranging from small protein assemblies to organelles. 

Each term (ie. molecular pathway) in the NeST hierarchy is embedded with a unique fully connected linear layer 

# Challenge Goals

## Optimize VNN model
Implement a sweep with W&B to search a defined hyperparameter space and identify better performing hyperparameter combinations

While there are many potential hyperparameter configurations, there is not enoguh time in this training event to perform any extensive tuning. A better approach may be to choose one or two of the following hyperparameters and sweep a limited range of values. Possible choices might include:

* Regularization
  * Dropout
  * Dropout layer
  * Weight decay
  * Early stop function
  * Adaptive learning rate
  * Optimization method

* Architecture
  * Activation function
  * Loss function
  * Neurons per layer

* Objective
  * Predicting probability of a response instead of AUDRC

## Evaluate predictive performance of drug response models
Measuring performance of drug response models can be done in several ways, and care should be taken when selecting a metric. When predicting drug responses, the model naturally creates an ordered list of samples ranging from most to least likely to respond to therapy. In our case, the model will predict the continuous value AUDRC, so it might seem logical to use euclidean distance measurements such as mean squared error. However, it is generally more helpful to identify the tumors most/least likely to respond, so correlation (eg. Pearson, Spearman) is generally used. Performance can also be measured categorically, as is the case with various receiver/operator statistics (eg. precision-recall). For continuous value predictions, these categorical metrics establish a threshold prediction value, then consider a sample sensitive/resistant if the prediction lies below/above the threshold. Ultimately, the metric selected must fit the task at hand. 
