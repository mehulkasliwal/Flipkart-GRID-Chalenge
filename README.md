# Flipkart-GRID-Chalenge
## Flipkart_Grid - 2019 - Team Name: AInsteins

The basic idea of object detection using bounding box is to detect a real life
independent object by specifying coordinates of bounding box for the object.

### Pre-Processing :

The images were pre-processed by the following steps :

1. To scale the RGB values from [0, 255] to [-1, 1], the values were subtracted from
    mean and divided by standard deviation.
2. Next we have grayscale the images , as they are more suitable and also reduces
    the amount of features by reducing the total no of data channels used from three
    to one, which would roughly reduce the size of the data by 3.
3. Finally the last step of preprocessing include reshaping the size of the image to a
    suitable value such that training it on a large network is feasible within
    reasonable bounds of data loss. The dimensions that we chose was 192*
    pixels ,reshaped from the original picture of size (640*480). We also tried
    modelled the data using smaller dimensions of 64*64, 128*128 , however due to
    higher data losses , the model suffered from lower accuracies. Also ,now that we
    have reshaped the images we made suitable changes to our xmin,x,ymin,ymax
    in the training set by multiplying them with the requisite scaling factor.

### Training :

A sequential deep learning model was designed keeping in mind the complexity and
volume of data to be learned. The block diagram on the next page summarizes the
architecture of the whole neural network.
![Model1](https://github.com/mehulkasliwal/Flipkart-GRID-Chalenge/blob/master/Model1.jpg)
![Model2](https://github.com/mehulkasliwal/Flipkart-GRID-Chalenge/blob/master/Model2.jpg)

### Features Used :

● EarlyStopping was used with a minimum threshold of 0.0001 for delta to prevent
model from overfitting.  
● Callbacks feature of Keras was used to store the timesnap of model weights after
epoch.  
● Adam was chosen as optimizer for the model because of its excellent record with
regression data.  
● MSE loss function was used to calculate the deviation from true value of the
predicted value.  

### Testing :

With the model trained , we finally put it to test with 12800 test samples. Similar to the
Training Model, we first reshape the size of the image to 192*192 pixels. Next, we load
the weights trained by the model previously on the training set. Next , we start predicting
the output and finally export the predictions in the corresponding csv format.
Our model reported an accuracy of ~88%.


