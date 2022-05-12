# DLFinalProject
Liver Tumor Segmentation for Improving Visualization and Treatment Outcomes
Abigail Marks, Priyanka Solanky,Ria Panjwani

Introduction
According to the CDC, each year in the United States, 24,500 men and 10,000 women get liver cancer, and 18,600 men and 9,000 women die from the disease. As physicians diagnose patients, they often utilize Magnetic Resonance Imaging (MRI) to visualize liver tumors from tomography images. Analyzing structural MRI scans is a time-consuming task, so automatic and robust liver tumor segmentation can support radiologists glean information about size, shape to enhance personalized treatment plans. Deep learning models, such as the one we want to implement, have had great success with medical segmentation tasks, allowing physicians  to visualize tumors to enhance treatment options, thus improving overall patient outcomes. 
	The paper we are implementing describes automatic brain tumor detection and segmentation, using U-Net based fully convolutional networks. The paper’s objective is to provide brain tumor segmentation that is accurate and automatic. High accuracy allows for precise classification of tumor subtypes and border delineation, both useful in surgical planning, and in tracking tumor growth/shrinkage. We chose this paper because we find this application exciting and extremely pertinent to real life outcomes. Being able to replicate the architecture this paper outlines has the potential to save lives, money, and time. We have also found from personal experience that finding large and robust datasets has been a challenge, so we looked at many different projects and felt comfortable working with medical imagery data. As group members, we all found the project attainable given our understanding of deep learning, relevant to the real world with a tangible practical  application, and a challenge for us to work through together. 
	This segmentation task is enhancing the representation of the tumor within an image into something more meaningful and easier to analyze. In this way, segmentation allows for automatic identification of the location of liver tumors within the liver, and clear boundaries of the edges of the tumor. This task is a problem of its own, as it is not quite classifying images as done with MNIST, nor is it any sort of prediction task. 
 
Methodology
First we will need to preprocess our data. We are using .nii files that come from MRIs. The MRIs are of the whole body and not just areas involving the liver, so we will need to get our images into the right form. They also are in 3 dimensions. We will have to slice the data on an axis to get 2D images. They also will have to be a reasonable size given that convolution is not a computationally light task to complete. If needed, we will also perform some data augmentation to help our accuracy.
We will be training a U-Net model for segmentation. A U-Net is a fully convolutional model that first extracts the features of the image with a series of convolution and max pooling layers. Once it has downsample to a certain point, it begins to upsample again. It uses skip connections to enhance the context of the upsampling.  This is a form of supervised learning, so we will train the model over a series of epochs and in batches. The structure of the model will be as follows:
Layer 1: Image size 240x240 
Conv 2d 3x3, 64 filters,
Conv 2d 3x3, 64 filters
Save output for later
Max pooling 2x2
Layer 2: Image size 120x120 
Conv 2d 3x3, 128 filters,
Conv 2d 3x3, 128 filters
Save output for later
Max pooling 2x2
Layer 3: Image size 60x60
Conv 2d 3x3, 256 filters,
Conv 2d 3x3, 256 filters
Save output for later
Max pooling 2x2
Layer 3: Image size 30x30
Conv 2d 3x3, 512 filters,
Conv 2d 3x3, 512 filters
Save output for later
Max pooling 2x2
Layer 3: Image size 30x30
Conv 2d 3x3, 512 filters,
Conv 2d 3x3, 512 filters
Save output for later
Max pooling 2x2
	Layer 4 (bottom of the U)
Con 2d 3x3, 1024 filters
Conv 2d 3x3 1024 filters
Deconv 3x3
	Layer 5
copy and concatenate saved output from layer 3
conv 2d 3x3 256 filters
conv 2d 3x3  256 filters
deconv 3x3
	Layer 6
copy and concatenate saved output from layer 2
conv 2d 3x3 128 filters
conv 2d 3x3  128 filters
deconv 3x3
Layer 7
copy and concatenate saved output from layer 1
conv 2d 3x3 64 filters
conv 2d 3x3  64 filters
conv 2d 1x1 2 filters
	The paper uses an Adam optimizer with a learning rate of .001. The weights will be initialized at random normal with mean 0 and standard deviation .01. They initialized all biases to 0. The architecture above is what was given in the paper. However, due to our small amount of data and compute resources, we ultimately scaled down the model. This led to better results and less overfitting. Instead of starting with 64 filters in the first layer, we start with 4. We experimented with both, and ultimately found better results with the smaller model. We used soft dice loss, which is calculated by 1-2|A∩B|+1|A|+|B|+1. We experimented with different loss functions, including 1-precision and 1- IOU. The best results were with the soft dice score.
Results:
We ended up with a model with 91% accuracy on the train set and 89% accuracy on our test set. Accuracy was measured by Intersection Over Union. The IOU accuracy is calculated by the area of overlap of the segmentations divided by the union of the segmentation areas. In terms of pixel accuracy, we achieve over 99% accuracy. However, pixel accuracy is a bad way to measure the accuracy so it was not our standard. Our model performs better on larger well defined tumors than on smaller ones. On smaller tumors, sometimes the model does not pick up on the fact that the tumors exist. On large tumors, the model shows high accuracy. This could be because of an unbalanced training set. If the dataset has more large tumors than small ones, the model will learn how to detect those better.
Challenges
There were a few major challenges in implementing our paper. The first challenge was running our model quickly. There are many layers and many computations, so the model took a really long time to run. To solve this problem we decided that running our code on GPU powered platforms would be more efficient. We decided to work mostly in a Colab notebook and use the GPU runtime. However, we would run into the GPU limits and RAM limits. It also meant that we could not train the model on all of the data that we had because there was not enough available memory to have all of the filters, images, and layers needed. We tried using GCP, but found it too difficult to make small changes to the model by only interacting with the code via nano in the SSH terminal.
The second challenge was data; the .nii files we were using take up a lot of space. Downloading all of the data takes up a lot of space on our machines and leads to difficulties. We had to process the data in batches in order to be able to use it all. Originally, we were only using 50 of the .nii files, which led to around 1400 images. With this amount of data, even after reducing the number of parameters in the model, it was still getting extremely overfit. However, after processing the data in batches, saving it in a compressed format, then deleting the original larger files allowed us to download around 80 more .nii files. These files led to an additional 6000 images. However, we ran into troubles with this amount of data as well. Lastly, as mentioned briefly above, we struggled with our model overfitting because we did not have enough data.
Reflection 

The project was successful as we achieved a high accuracy. We did well according to our base, target and stretch goals. The model did work the way that we expected it to work as we followed the U-Net deep convolutional network architecture described in the paper. The approach changed over time as our model originally started with 64 filters in the first layer which caused issues as there was not enough data to support 64 layers and thus too many trainable parameters which led to overfitting of the data. In order to scale back the model, we divided the number of filters in the first layer to have 4 in the first layer. This drastically reduced the number of trainable parameters and solved the overfitting issue. We also adjusted the loss function, as initially there were too many false positives, switching the loss function to be 1-precision. However, this led to the model not recognizing enough positive cases. Ultimately, this issue was due to the lack of data and not the loss function and so we reverted back to using the soft dice loss function described in the paper.  We would take these factors and knowledge into consideration immediately if we could do this project again. If we had more time, we would run the model with more data. We would also scale up the number of parameters as we have more data now and thus it is possible that our model would do better with more layers. It would also be helpful to explore if our model could handle segmentation with kidneys or other organs as well to detect tumors to see how general our model is. From this project, we have learned the difficulty of working with large quantities of data and running it through a model. We also gained confidence in applying deep learning algorithms in order to apply segmentation to livers and detect tumors. 
