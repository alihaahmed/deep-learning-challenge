# deep-learning-challenge

## Overview of the Analysis
The nonprofit foundation Alphabet Soup is in search of a tool that can enable it to select applicants for funding with the highest probability of success in their ventures. This report discusses the creation and subsequent optimization process of a neural network that is capable of performing binary classification with a target predictive accuracy of at least 75% to predict whether applicants will be successful if funded by Alphabet Soup.

A CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years was read in to a Pandas DataFrame using Google Colab. Within this dataset are a number of columns that capture metadata about each organization, such as:
* EIN and NAME — Identification columns
* APPLICATION_TYPE — Alphabet Soup application type
* AFFILIATION — Affiliated sector of industry
* CLASSIFICATION — Government organization classification
* USE_CASE — Use case for funding
* ORGANIZATION — Organization type
* STATUS — Active status
* INCOME_AMT — Income classification
* SPECIAL_CONSIDERATIONS — Special considerations for application
* ASK_AMT — Funding amount requested

## Steps 

**1. Data Preprocessing**
All necessary dependencies were first imported (train_test_split from sklearn.model_selection, StandardScaler from sklearn.preprocessing, Pandas and Tensorflow). Upon reading in the CSV file to a Pandas DataFrame, two identification columns ('EIN' and 'NAME') were determined to be non-beneficial and thus dropped. The 'IS_SUCCESSFUL' column was determined to be the target variable for the model, with all other 9 columns/variables representing input feature variables:

* APPLICATION_TYPE
* AFFILIATION
* CLASSIFICATION
* USE_CASE
* ORGANIZATION
* STATUS
* INCOME_AMT
* SPECIAL_CONSIDERATIONS
* ASK_AMT

The number of unique values in each column were determined using .nunique(). Two categorical features, 'APPLICATION_TYPE' and 'CLASSIFICATION', were found to have more than 10 unique values with 17 and 71 unique values, respectively. A cutoff value was chosen for each variable for binning purposes using .value_counts(), with all rare values below the cutoff threshold bucketed into an "Other" category. All categorical data was then numerically encoded using pd.get_dummies within the DataFrame. The preprocessed data was split into feature and target arrays and subsequently split into a training and testing dataset. A StandardScaler instance was initiated, only the training feature data was fitted to prevent leakage in the workflow, and both training and test feature data were scaled and transformed.


**2. Compiling, Training and Evaluating the Model**

**Preliminary Model**:
  * Input features: 43
  * Total neurons: 90
  * Total hidden layers: 2
  * Neurons per hidden layer: 50, 40
  * Hidden layer activation function(s): ReLU
  * Output layer activation function: Sigmoid
  * Output node: 1
  * Epochs: 100
  * Evaluation: 55.33% loss and 72.58% accuracy rate on test data
  * Discussion: A "middle ground" approach was taken with the first model, with roughly twice as many total neurons as input features and two hidden layers. Given our binary classification use case, a Sigmoid activation      function was employed in the output layer. Neurons were roughly evenly split between each hidden layer. At an overall accuracy of 72.58% and 55.33% loss rate, there is substantial room for improvement and target          model performance has not yet been achieved.

**3. Optimizing the Model**

**Optimization Model 1:**
  * Changes implemented in preprocessing stage: Increased cutoff value for binning 'APPLICATION_TYPE' column from 500 to 700
  * Input features: 43
  * Total neurons: 110
  * Total hidden layers: 3
  * Neurons per hidden layer: 60, 30, 20
  * Hidden layer activation function(s): ReLU
  * Output layer activation function: Sigmoid
  * Output node: 1
  * Epochs: 100
  * Evaluation: 55.94% loss and 72.57% accuracy rate on test data
  * Discussion: In the preprocessing stage, the binning process was slightly improved by increasing the cutoff value for the 'APPLICATION_TYPE' column from 500 to 700. The first optimization model increased the total 
    neuron count slightly from 90 to 110 and one more hidden layer was added to allow for more processing, which may enable the model to capture more intricate patterns and interactions within the data to help                enhance accuracy. Activation functions in the hidden layers and output layer were maintained. As a result of these changes, model performance actually deteriorated slightly with the loss rate increasing by 0.61% and      accuracy rate decreasing by 0.01%.
 
**Optimization Model 2:**
  * Input features: 43
  * Total neurons: 45
  * Total hidden layers: 4
  * Neurons per hidden layer: 20, 10, 10, 5
  * Hidden layer activation function(s): ReLU
  * Output layer activation function: Sigmoid
  * Output node: 1
  * Epochs: 120
  * Evaluation: 55.35% loss and 72.48% accuracy rate on test data
  * Discussion: A different approach was taken with the second optimization model and a deeper neural network was implemented by increasing the hidden layers to 4. Since increasing the neuron count in the previous model      did not seem to enhance performance at a surface level, the neuron count in this model was reduced to maintain a rough 1:1 ratio with the number of input features. In addition, neurons were more evenly spread out         across hidden layers as opposed to concentrating roughly half in the first hidden layer, as was done previously. The number of epochs was also increased from 100 to 120 to slightly prolong the iterative process while     balancing against the risk of overfitting the training data. As a result of these changes, model accuracy further deteriorated with a -0.09% change from the previous model. It may be likely that the neuron count          should have been kept in the more appropriate and standard range of 2-3x the number of input feature variables.

  **Optimization Model 3:**
  * Input features: 43
  * Total neurons: 130
  * Total hidden layers: 2
  * Neurons per hidden layer: 70, 60
  * Hidden layer activation function(s): ReLU
  * Output layer activation function: Sigmoid
  * Output node: 1
  * Epochs: 100
  * Evaluation: 55.64% loss and 72.55% accuracy rate on test data
  * Discussion: Model accuracy was brought back closer in line with earlier optimization attempts by drastically increasing the number of neurons to 130 and decreasing the number of hidden layers to 2. The number of          epochs was also brought back down to 100; from the epoch loss and accuracy data plotted within the notebooks, it appears that loss and accuracy rates begin to plateau around the 80-epoch mark and thus the risk of         overfitting the training data by increasing the number of epochs beyond this point becomes significant. However, the 75% predictive accuracy target has still not been achieved even after three manual optimization         attempts. It is likely that more drastic tweaks need to be made, including implementing a different activation function in the hidden layers. To further explore this, an automated hyperparameter optimization/tuning       framework was implemented using the keras-tuner library to streamline the trial and error process. A function was built to create a new Sequential model with a range of hyperparameter options allowing kerastuner          to seek and determine the optimal combination of hyperparameters to optimize for validation accuracy. The hyperparameter ranges provided to this function and the results of the kerastuner search are as follows:
  
**Automating Hyperparameter Tuning With Keras Tuner:**
  * Function set-up and parameters
      * Input features: 43
      * Activation choices: ReLU, Tanh
      * Neurons in first layer: 1-50 in steps of 5
      * Number of hidden layers: 1-5
      * Neurons in hidden layers: 1-40 in steps of 5
      * Output layer activation function: Sigmoid
      * Objective: Validation accuracy
      * Maximum epochs: 50
      * Hyperband iterations: 2
  * Total elapsed time in Keras Tuner search: 00h 27m 06s
  * Top three models hyperparameter values
      * Model 1: {'activation': 'tanh', 'first_units': 41, 'num_layers': 4, 'units_0': 21, 'units_1': 11, 'units_2': 6, 'units_3': 11, 'units_4': 21, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 2, 
       'tuner/round':          2, 'tuner/trial_id': '0038'}
      * Model 2: {'activation': 'relu', 'first_units': 26, 'num_layers': 2, 'units_0': 16, 'units_1': 21, 'units_2': 16, 'units_3': 36, 'units_4': 16, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0,             'tuner/round': 0}
      * Model 3: {'activation': 'tanh', 'first_units': 6, 'num_layers': 3, 'units_0': 31, 'units_1': 36, 'units_2': 11, 'units_3': 21, 'units_4': 6, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 1, 
       'tuner/round': 
         1, 'tuner/trial_id': '0046'}
  * Top three models evaluation against test data
      * Model 1: 55.30% loss, 72.94% accuracy
      * Model 2: 55.09% loss, 72.89% accuracy
      * Model 3: 55.67% loss, 72.87% accuracy

## Summary
Of the three manually optimized models, the very first non-optimized model delivered the strongest loss and accuracy metrics with an accuracy rate of 72.58%. Subsequent optimization attempts fell short of this figure, with accuracy fluctuating in a very narrow range of 72.48%-72.57% across all three manual attempts. The results of the kerastuner search were interesting for many reasons, the first being that all three models determined from the search fell short of the 75% predictive accuracy target. As a matter of fact, all three models revealed by the kerastuner search improved accuracy by just less than 1% relative to the manual optimization attempts made. Moreover, the hyperparameter values and depth of each model vary substantially from one another; two of the three top models utilize a tanh activation function and each model also consists of a different number of hidden layers. As such, it appears that a Tanh activation function in the hidden layers is most appropriate to address the performance issue and classification problem. Overall, these inconclusive results suggest that more adjustments need to be made in the data preprocessing stage to meaningfully optimize the model for accuracy. In particular, the shape of the dataset, determining the appropriate balance of input features and eliminating noise/outliers in the data should result in meaningful changes in model performance when simply adjusting the model parameters themselves doesn't appear to do the trick.
