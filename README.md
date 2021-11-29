# Mwdb-Project-Phase-3

To install the packages use
`pip install -r requirements.txt`.

To run the project use
`python main.py`. Works with only python3.

```
Outputs:
    1. The .csv files which contantis the semantics will be stored in the folder called "output".
    2. ALl the information is displayed in the console only.
    3. The outputs for the task 4 are stored in lsh_output.
    4. The outputs for the task 5 are stored in Output_VAFiles
    5. The outputs for the task 6 and 7 are stored in the older <Classifier>_feedback_output_task_<pre requisite task number>
```


### Workflow of application
The main.py has the code which handles all tasks. The first input to the program will be the task number.
For each task the appropriate inputs are asked to get the output after performing the specified tasks.





###### Task to create the latent semantics using all the images

#### Task 1, 2, 3

```
Dataset: Dataset input to train or create the classifier model. This acts as the traing dataset.
Test Image Path: The test images path for classification.
Feature: The feature method to extract the features of the images - Color moments, ELBP, HOG
k: The value of k to perform the dimentionality reduction. Can take a string value "all" to use the original features to train.
classifier: To use which classifier to classify the test data. SVM, Decision-Trees, PPR.
```
The inputs for the three tasks are same and they only differ in the labels for each image in the dataset.

#### Task 4, 5
Common inputs for both task 4 and 5:
```
Dataset path: The dataset path to be used for LSH and VA Files.
Test Image Path: The test images path for classification.
Test Image name: The test image name to retrieve the t similar images.
Feature: The feature method to extract the features of the images - Color moments, ELBP, HOG
```

Inputs for task 4:
```
Layers: The number of layers in LSH.
Hashes: The number of hashes required per layer.
t: The number of similar images needs to be retrieved.

Outputs will be stored in lsh_output folder.
```

Inputs for task 5:
```
t: The number of similar images needed to be retrieved.
b: The number of bits to be used in VA Files.
```

#### Task 6, 7
```
These tasks are run after tasks 4 and 5 are performed which is a pre-requisite.

Inputs for these tasks are retrieved from the output of task 4 or 5
Outputs will be stored in the folder {classifier}_feedback_output_for_task_{4/5}.

Task 6 alone takes value of 'k' to perform the dimentinality reduction since decision tree has heavy computations.
```

#### Project Structure.

The main.py is the start of the code.
The folder tasks contains the code for each tasks and supporting code functions.
The output folder contains the .csv of latent semantics.
The lsh_output contains the results of task 4
The Output_VAFiles contains the results of task 5
The decion_tree_feedback_output_for_task_4/5 contains the outputs for task 6 when ran after task 4 or 5
The svm_feedback_output_for_task_4/5 contains the outputs for task 7 when ran after task 4 or 5
