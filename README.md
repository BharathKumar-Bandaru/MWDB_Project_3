[# Mwdb-Project-Phase-2

To install the packages use
`pip install -r requirements.txt`.

To run the project use
`python main.py`. Works with only python3.

```
Outputs:
    1. The .csv files which contantis the semantics will be stored in the folder called "output".
    2. The n similar images for task 5 will be stored in folder called "output2".
    3. The other necessary information will be displayed in the console.
```


### Workflow of application
The main.py has the code which handles all tasks. The first input to the program will be the task number.
For each task the appropriate inputs are asked to get the output after performing the specified tasks.



#### Task 0 

###### Task to create the latent semantics using all the images

When asked for task number type `0`
```
Enter the Dataset path : Default_value = "Dataset". This is where images are stored for processing.
Enter the value of feature: The feature extract method.
Enter the value of k: Should be an integer and less than the no.of features in the feature extract menthod that is used.
    If the value of k is more then the feature dimention the program will ask for the input again.
Enter the value of dimentionality Reductoin: Dimentionality reduction technique: values = [pca, svd, lda, kmeans]

Output will be the latent semantic files in .csv format. 
```

#### Task 1
When asked for task number type `1`

```
Enter the Dataset path : Default_value = "Dataset". This is where images are stored for processing.
Enter the value of feature: The feature extract method.
Enter the value of k: Should be an integer and less than the no.of features in the feature extract menthod that is used.
    If the value of k is more then the feature dimention the program will ask for the input again.
Enter the value of dimentionality Reductoin: Dimentionality reduction technique: values = [pca, svd, lda, kmeans]
Enter the type of the images you want to perform the task1.

Output will be the latent semantic files in .csv format. 
The subject weights of each latent semantics are displayed in console output.
```

#### Task 2
When asked for task number type `2`

```
Enter the Dataset path : Default_value = "Dataset". This is where images are stored for processing.
Enter the value of feature: The feature extract method.
Enter the value of k: Should be an integer and less than the no.of features in the feature extract menthod that is used.
    If the value of k is more then the feature dimention the program will ask for the input again.
Enter the value of dimentionality Reductoin: Dimentionality reduction technique: values = [pca, svd, lda, kmeans]
Enter the subject_id of the images you want to perform the task2.

Output will be the latent semantic files in .csv format. 
The type weights of each latent semantics are displayed in console output.
```

#### Task 3
When asked for task number type `3`

```
Enter the Dataset path : Default_value = "Dataset". This is where images are stored for processing.
Enter the value of feature: The feature extract method.
Enter the value of k: Should be an integer and less than the no.of features in the feature extract menthod that is used.
    If the value of k is more then the feature dimention the program will ask for the input again.
Enter the value of dimentionality Reductoin: Dimentionality reduction technique: values = [pca, svd, lda, kmeans]

Output will be the .csv file which stores the latent semantics, right factor matrix and similarity matrix
The type weights pairs of each latent semantics are displayed in decreasing order.  
```

#### Task 4
When asked for task number type `4`

```
Enter the Dataset path : Default_value = "Dataset". This is where images are stored for processing.
Enter the value of feature: The feature extract method.
Enter the value of k: Should be an integer and less than the no.of features in the feature extract menthod that is used.
    If the value of k is more then the feature dimention the program will ask for the input again.
Enter the value of dimentionality Reductoin: Dimentionality reduction technique: values = [pca, svd, lda, kmeans]

Output will be the .csv file which stores the latent semantics, right factor matrix and similarity matrix
The subject weights pairs of each latent semantics are displayed in decreasing order.  
```

#### Task 5
When asked for task number type `5`

```
Enter the image path: Dataset/<Image_name>. Eg: Dataset/image-cc-1-1.png
Enter the path for latent semantics: output/<Sematic_file>. Eg: output/cm_svd_latent_semantics.csv
Enter the value of n to retrieve the n similar images.
Enter the feature value that of the latent semantics.

Output will generate top n similar files and stores them in output2. 
```

#### Task 6
When asked for task number type `6`

```
Enter the image path: Dataset/<Image_name>. Eg: Dataset/image-cc-1-1.png
Enter the path for latent semantics: output/<Sematic_file>. Eg: output/cm_svd_latent_semantics.csv
Enter the feature value that of the latent semantics.
Enter the dimentionality reduction of the latent sematnics.

Output will generate type lable to which the input image belongs to. 
```


#### Task 7
When asked for task number type `7`

```
Enter the image path: Dataset/<Image_name>. Eg: Dataset/image-cc-1-1.png
Enter the path for latent semantics: output/<Sematic_file>. Eg: output/cm_svd_latent_semantics.csv
Enter the feature value that of the latent semantics.
Enter the dimentionality reduction of the latent sematnics.

Output will generate subject id to which the input image belongs. 
```

#### Task 8
When asked for task number task `8`

```
Enter the similarity matrix path: output/<semantic_file>. Eg: output/cm_svd_latent_semantics.csv
Enter the value of n.
Enter the value of m.

Output displays the n most similar subjects.
```

#### Task 9
When asked for task number task `9`

```
Enter the path of similarity matrix: 
Enter the value of n.
Enter the value of m.

Output will be top subject id's
```


#### Project Structure.

The main.py is the start of the code.
The folder tasks contains the code for each tasks and supporting code functions.
The output folder contains the .csv of latent semantics.
The output2 folder contains the output images of task 5.
