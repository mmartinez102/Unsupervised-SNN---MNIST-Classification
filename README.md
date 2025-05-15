# Unsupervised-SNN---MNIST-Classification
A simple implementation of the unsupervised SNN model created by Peter Diehl and Matthew Cook. It trains and tests an unsupervised SNN using the MNIST dataset. Still needs work to speed up calculations. 

To allow the MNIST dataset to be accessible to the code, unzip the MNIST file and save the whole folder (with the name "MNIST" in your C drive).

The Diehl model with sparsity employs a slight modification to the input spike train creation, where one can force sparsity on the input spike trains. This allows the images to create less spikes, and gives a fair chance for all classes to have some representation among the excitatory neurons when it comes to assigning them to classes. 
