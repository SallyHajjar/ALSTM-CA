# Readme
Our article is titled "Long short term memory and attention models for simulating urban densification"
This correspondence addresses urban densification
It presents a novel cellular automata model using a neural network approach that combines the Long short term memory,
the attention, and the neural network models
It is developed, calibrated, and verified for Belgium
Performance is compared with some state-of-the-art techniques
The data will be given upon request
The file "mainmodel" describe the proposed LSTM-Attention CA model

## Installation
Google Colab was used for validation. Colab allows writing and executing arbitrary python code through the browser and is especially well suited to machine and deep learning.
More technically, Colab is a hosted Jupyter notebook service that requires no setup to use, while providing access free of charge to computing resources including GPUs.
First, the file "Librairies.txt" should be executed to install all the functions available in Colab and needed in our project.
## Definition of all functions
This file includes the definition of all non-built-in functions. It also includes the definition of the function used to compute the effects of neighboors on a defined cell.
## Data Pre-processing
The folder data contains all the raster data used in our paper.
We have used three (100x100) m raster-based built-up maps for 2000, 2010, and 2020 for Belgium. Moreover, 17 raster files, corresponding to different driving factors, are also pre-processed and used as inputs to our method. To process these datasets, we use the file called "Pre-processing.txt".
We also used some functions defined in the file"Functions'.txt. The neighborhood statistics are also considered as inputs to our model.
## Models
Our model is a novel cellular automata model using a neural network approach that combines the Long short term memory, the attention, and the neural network models.
###### Attention model
First, we should execute the file called "Attentionmodel.txt".
###### Attention + LSTM + NN models + Results
Then, we execute the file "mainmodel.txt".
###### Comparison with SVM and MLR
The file called "SVM&MLR" contains the competing models. We execute this file and compare the obtained results with our model.
## Testing the proposed model
To test the final model an example is given in the folder called test, which was tested using Google Colab.
