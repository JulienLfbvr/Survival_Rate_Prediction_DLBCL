# Survival_Rate_Prediction_DLBCL

M1 project, which consists in the development of an AI model that predicts the survival rate of patients suffering from DLBCL on the basis of histopathological images.  

## Abstract 

The aim of this project was to predict the survival rate of patients with DLBCL by extracting relevant features from histopathological images using AI and, more specifically, CNNs. To build our dataset, we then combined these features with the patients' clinical data. We then used various methods such as PCA, UMAP or simply selecting the variables with the highest variances to try and reduce the number of features extracted. Once we had selected the features, we fitted a Cox proportional hazards model in order to demonstrate the usefulness of these extracted features in predicting the outcome of survival.

## Dataset 

DLBCL-Morph is publicly available at this link: 
https://stanfordmedicine.box.com/s/ub8e0wlhsdenyhdsuuzp6zhj0i82xrb1

this dataset has been made available thanks to : 
@misc{vrabac2020dlbclmorph,
    title={DLBCL-Morph: Morphological features computed using deep learning for an annotated digital DLBCL image set},
    author={Damir Vrabac and Akshay Smit and Rebecca Rojansky and Yasodha Natkunam and Ranjana H. Advani and Andrew Y. Ng and Sebastian Fernandez-Pol and Pranav Rajpurkar},
    year={2020},
    eprint={2009.08123},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

## Model 

In order to build our model, we have fine-tuned another model called KimiaNet (https://arxiv.org/pdf/2101.07903v1.pdf). KimiaNet itself is a fine-tuned version of Densenet121 that has been trained on histopathological images of different types of cancer, but not on DLBCL. To fine-tune this model, we built an autoencoder that takes H&E stained pathces as input. We used KimiaNet as the encoder part and built the decoder based on the Densenet121 architecture. The idea behind using an autoencoder is to perform unsupervised training of a model using the input as the output. Convolutional layers in the encoder reduce the dimensions of the input images and then the decoder tries to reconstruct the original images using a lower dimensional representation of the input images. In this way, we want to understand and represent only the deep correlations and relationships between data.

## Code usage
