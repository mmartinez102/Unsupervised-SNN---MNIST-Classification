#This is an implementation of the Peter U.Diehl and Matthew Cook unsupervised SNN model.
#We can select how many images to train on from the MNIST training dataset. We can also choose how many to test on as well.
#We also try to set the parameters to the proper units (V, S, s, A, etc)
#Three sections exist in this code. First section applies STDP and intrinsic plasticity learning to the SNN during the presentation of each MNIST image.
#The image is presented for 350 ms, and then taken away for 150 ms to allow the dynamical variables to fall back to steady-state values.
#During the second section, the same training images are fed back into the SNN but with a learning rate set to zero (no STDP) an a fixed membrane potential threshold (no instrinsic plasticity).
#The max firing rates are recorded for each image for each excitatory neuron, and then each neuron is assigned a class based on which one it fired the most for.
#In the third section, we test the network by feeding in an adjustable number of training images to the SNN (with STDP and intrinsic plasticity still disabled) and record
#the classification accuracy. We also run a random number generator to see how the result fairs against random classifiying. 

import numpy as np
import matplotlib.pyplot as plt
import struct 
from datetime import datetime

start_time = datetime.now()

#Setting up the SNN architecture
N = 100  #Number of excitatory neurons (same as inhibitory neurons)
I = 784  #Number of input neurons (28x28 for MNIST)
O = 10   #Number of output neurons (for 0-9 digit classification)

#LIF model parameters
Erest = -65 * (1e-3)  #Resting membrane potential (V)
Eexc = -50 * (1e-3)   #Excitatory reversal potential (V)
Einh = -70 * (1e-3)   #Inhibitory reversal potential (V)
Vth_base = -25 * (1e-3)  #Base threshold membrane voltage for spiking

# Time Constants
taue = 0.005   #Excitatory neuron membrane time constant (s). Working value is 0.002
tauge = 0.004   #Excitatory conductance time constant (s). Working value is 0.001
taugi = 0.007 #Inhibitory conductance time constant (s). Decreasing this parameter changes the number of excitatory neuron spikes 0.005 
ratio = taugi/tauge #Ratio of inhibitory to excitatory conductances. Working value is 

#Simulation Time Parameters
T_max = 0.35  #Total simulation time with spikes (s)
T_none = 0.15 #Total simulation time with no input spikes (s)
T_total = T_max + T_none  #Total simulation time (500 ms)
dt = 0.001     #Time step (s)
num_steps = int(T_total / dt) #Total number of timesteps 
MNIST_size = 100 #Number of images to train on in MNIST. 100 works.
MNIST_test_size = 1000 # Number of test samples (usually 10,000 for MNIST)
MNIST_repeat = 1 #Nummber of times we present the MNIST dataset of size above

#Spike vectors for input neurons, excitatory neurons, and inhibitory neurons are initialized here
SpikeIn = np.zeros((num_steps, I))
SpikeExc = np.zeros((num_steps, N))
SpikeInh = np.zeros((num_steps, N))

#Membrane potential for excitatory neurons is initialized
Vexc = np.full((num_steps, N), Erest)

#Conductances initialized for both inhibitory and excitatory synapses
ge = np.zeros((num_steps, N), dtype=np.float32)
gi = np.zeros((num_steps, N), dtype=np.float32)

#Synaptic weights initialized
wExc = np.random.rand(N, I).astype(np.float32) #Weights from input neurons to excitatory neurons
wIE = (np.random.rand(N, N)).astype(np.float32) #Weights from inhibitory neurons to excitatory neurons
np.fill_diagonal(wIE, 0)  #Makes zero diagonal indices
print(f"Pretrained weights wExc: {wExc}")
# print(f"Pretrained weights wIE: {wIE}")

#Presynaptic trace parameters setup
Xpre = np.zeros((num_steps, N, I))
taupre = 0.005  #Decay constant. 0.001 was the working value
Xtar = 2  #Target trace value: was 2 before. 
eta = 1e-6 #Learning rate: 5e-8 - 1e-1 gives 14% acc. 1e-5 gave the most diverse classes with bad 7.6% acc. 8e-6 also diverse.
wmax = 0.99  #Max weight
wmin = 0.01 #Minimum weight
mu = 1  #Power law exponent
#To avoid repeating this calculation in the Xpre side:
exp_taupre = np.exp(-dt / taupre)

#Intrinsic plasticity parameters setup - Used to make fair balance of input to neurons spiking many times
tau_theta = 0.02  #Increasing this parameter reduces the number of spikes of the excitatory neurons. 
gamma = 0.75 # Reducing this parameter increases the number of spikes of the excitatory neurons. 
theta = np.zeros((num_steps, N))  
Vth_exc = np.full((num_steps, N), Vth_base)  

#Loading and preprocessing of the MNIST dataset
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(num_images, rows, cols)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), np.uint8)

#Define file paths
mnist_path = "C:/MNIST/"
train_images_path = mnist_path + "train-images.idx3-ubyte"
train_labels_path = mnist_path + "train-labels.idx1-ubyte"
test_images_path = mnist_path + "t10k-images.idx3-ubyte"
test_labels_path = mnist_path + "t10k-labels.idx1-ubyte"

#Load MNIST data
X_train = load_mnist_images(train_images_path)
Y_train = load_mnist_labels(train_labels_path)
X_test = load_mnist_images(test_images_path)
Y_test = load_mnist_labels(test_labels_path)

#Here is the training side of the network as it runs through all the images of the MNIST dataset that were selected.

# start_time = datetime.now()


for c in range(0, MNIST_repeat): #Loop through the MNIST dataset again MNIST_repeat times

    
    for k in range(0, MNIST_size): #Loop through all the MNIST images
        
        #We need to initialize the excitatory and inhibitory neurons in each run of new images.
        #Spike vectors for excitatory neurons, and inhibitory neurons are initialized here
        SpikeExc = np.zeros((num_steps, N))
        SpikeInh = np.zeros((num_steps, N))
        
        #Converting the MNIST mage to Poisson spike trains
        tr = k  #Image index
        image = X_train[tr] / 4

        # max_rate = 100 #Original value
        max_rate = 1
        time_window = int(T_max * 1000)  
        dtM = dt * 1000  

        firing_rates = image * max_rate
        spike_trains = np.random.rand(28, 28, time_window) < (firing_rates[..., None] * dtM / 1000)
        SpikeIn = spike_trains.reshape(28 * 28, time_window).T  
        SpikeIn = SpikeIn.astype(int)
        SpikeIn_none = np.zeros((int(T_none / dt), I))  #Create zero input for T_none time steps
        SpikeIn = np.vstack((SpikeIn, SpikeIn_none)).astype(bool)  #Concatenate the zero input period with the active input period

        for i in range(1, num_steps): #Loop through the time steps for each image or batch

            #Synaptic conductance calculations for both excitatory and inhibitory synapses
            ge[i] = ge[i-1] * np.exp(-dt / tauge) + wExc @ SpikeIn[i]
            gi[i] = gi[i-1] * np.exp(-dt / taugi) + wIE @ SpikeInh[i-1]

            #Membrane potential calculations 
            Vexc[i] = Vexc[i-1] + dt * ((Erest - Vexc[i-1]) + ge[i] * (Eexc - Vexc[i-1]) + gi[i] * (Einh - Vexc[i-1])) / taue
            Vexc[i] = np.maximum(Vexc[i], -70 * (1e-3)) #Keep Vexc from dropping below -70 mV
            
            #Excitatory Neuron Spike Detection & Reset
            spiking_neurons = Vexc[i] >= Vth_exc[i-1]
            SpikeExc[i, spiking_neurons] = 1
            Vexc[i, spiking_neurons] = Erest
            
            #When an excitatory neuron spikes, the inhibitory neuron its connected to fires.
            SpikeInh = SpikeExc

            #Intrinsic Plasticity (Threshold Adaptation)
            theta[i] = theta[i-1] * np.exp(-dt / tau_theta) + gamma * SpikeExc[i]
            Vth_exc[i] = Vth_base + theta[i]

            #Decay the presynaptic trace
            Xpre[i] = Xpre[i-1] * exp_taupre

            #Update presynaptic traces where spikes occurred
            Xpre[i][:, SpikeIn[i] == 1] += 1  

            #Compute STDP weight update in a batch operation IF an excitatory post synaptic neuron spikes
            spiking_indices = np.where(SpikeExc[i] == 1)[0]
            if len(spiking_indices) > 0:
                dw = eta * (Xpre[i][spiking_indices] - Xtar) * (wmax - wExc[spiking_indices]) ** mu
                wExc[spiking_indices] += dw 

            #Ensure weights remain within valid range
            np.clip(wExc, wmin, wmax, out=wExc)


# end_time = datetime.now()
# elapsed_time = end_time - start_time
# print(f"Execution Time: {elapsed_time}")


print(f"Trained weights wExc: {wExc}")
# print(f"Trained weights wIE: {wIE}")

















#Here is the training side of the network as it assigns each neuron a class (one of the 10 digits in the dataset) based on its highest response to the 10 digits
#when they are presented again (the same training dataset/size). In this round, the learning rate is set to zero and the intrinsic plasticity is turned off.
#We need to initialize the firing rate class vector for each neuron to store.
firing_class = np.zeros((N, O))

#We need to initialize the excitatory and inhibitory neurons in each run of new images.
#Spike vectors for excitatory neurons, and inhibitory neurons are initialized here
SpikeExc = np.zeros((num_steps, N))
SpikeInh = np.zeros((num_steps, N))
#Membrane potential for excitatory neurons is initialized
Vexc = np.full((num_steps, N), Erest)

# start_time = datetime.now()

for k in range(0, MNIST_size): #Loop through all the same MNIST images from training
    #We need to see which class/digit we are presenting in the kth iteration
    te = k  #Image index 
    label = Y_train[te] #Label gives us the label/digit of the image and the associated index that we will add the firing rates for each exc. neuron.
    
    #We need to initialize the excitatory and inhibitory neurons in each run of new images.
    #Spike vectors for excitatory neurons, and inhibitory neurons are initialized here
    SpikeExc = np.zeros((num_steps, N))
    SpikeInh = np.zeros((num_steps, N))
    
    #Converting the MNIST mage to Poisson spike trains
    tr = k  # Image index
    image = X_train[tr] / 4

    # max_rate = 100 #Original value
    max_rate = 1
    time_window = int(T_max * 1000)  
    dtM = dt * 1000  

    firing_rates = image * max_rate
    spike_trains = np.random.rand(28, 28, time_window) < (firing_rates[..., None] * dtM / 1000)
    SpikeIn = spike_trains.reshape(28 * 28, time_window).T  
    SpikeIn = SpikeIn.astype(int)
    SpikeIn_none = np.zeros((int(T_none / dt), I))  #Create zero input for T_none time steps
    SpikeIn = np.vstack((SpikeIn, SpikeIn_none)).astype(bool)  #Concatenate the zero input period with the active input period    
    
    for i in range(1, num_steps): #Loop through the time steps for each image or batch

        #Synaptic conductance calculations for both excitatory and inhibitory synapses
        ge[i] = ge[i-1] * np.exp(-dt / tauge) + wExc @ SpikeIn[i]
        gi[i] = gi[i-1] * np.exp(-dt / taugi) + wIE @ SpikeInh[i-1]

        #Membrane potential calculations 
        Vexc[i] = Vexc[i-1] + dt * ((Erest - Vexc[i-1]) + ge[i] * (Eexc - Vexc[i-1]) + gi[i] * (Einh - Vexc[i-1])) / taue
        Vexc[i] = np.maximum(Vexc[i], -70 * (1e-3))
        
        #Excitatory Neuron Spike Detection & Reset
        spiking_neurons = Vexc[i] >= Vth_exc[i-1]
        SpikeExc[i, spiking_neurons] = 1
        Vexc[i, spiking_neurons] = Erest
        
        #When an excitatory neuron spikes, the inhibitory neuron its connected to fires.
        SpikeInh = SpikeExc

        Vth_exc[i] = Vth_base
        
    #Check for excitatory spikes
    spike_counts = np.sum(SpikeExc, axis=0)
    
    #Creating matrix to assign classes (digits 0-9) to the excitatory neurons
    for l in range(0,N):
        if spike_counts[l] > firing_class[l][label]: 
            firing_class[l][label] = spike_counts[l]



# print(f"With no noise, firing_class: {firing_class}")
# #Add uniform noise between 0 and 10 to offer equal chances to the values that have close highest values
# firing_class += np.random.uniform(0, 10, size=firing_class.shape)
# print(f"After noise, firing_class: {firing_class}")

#Now we need to assign the excitatory neurons to the class with their highest firing rates per the firing_class matrix.
#This vector gives the class for each neuron.
#Normalize the neuron firing rates
firing_class_norm = firing_class / np.sum(firing_class, axis=1, keepdims=True)
# print(f"The firing_class_norm is: {firing_class_norm}")
# Assign neuron to class with the highest normalized response (prevents neurons from being biased to digits)
neuron_class = np.argmax(firing_class_norm, axis=1)
# neuron_class = np.argmax(firing_class, axis=1)
print(f"The classes for each excitatory neuron are as follows: {neuron_class}")

# end_time = datetime.now()
# elapsed_time = end_time - start_time
# print(f"Execution Time: {elapsed_time}")




















#Here is the testing side of the network as it classifies numbers. Each neuron's response (with its class) 
#The number it classifies to is the one that is selected
#In this round, the learning rate is set to zero and the intrinsic plasticity is turned off.

#This section is where we test out an image after training
#The response class is initialized as:
response_class = np.zeros(O)

#We need to initialize the excitatory and inhibitory neurons in each run of new images.
#Spike vectors for excitatory neurons, and inhibitory neurons are initialized here
SpikeExc = np.zeros((num_steps, N))
SpikeInh = np.zeros((num_steps, N))
#Membrane potential for excitatory neurons is initialized
Vexc = np.full((num_steps, N), Erest)

# Initialize correct predictions counter
correct_predictions = 0
total_predictions = 0  # Track total test cases processed
# Initialize correct predictions counter
correct_predictions2 = 0

# start_time = datetime.now()

# Generate a shuffled index
indices = np.random.permutation(len(X_train))
# Apply the shuffled indices to reorder the dataset
X_train = X_train[indices]
Y_train = Y_train[indices]


for k in range(0, MNIST_test_size): #Loop through all the same MNIST images from training
    
    # Reset response class for each image
    response_class = np.zeros(O)
    
    
    #We need to see which class/digit we are presenting in the kth iteration
    te = k  #Image index 
    label_test = Y_test[te] #Label gives us the label/digit of the image and the associated index that we will add the firing rates for each exc. neuron.
    
    #We need to initialize the excitatory and inhibitory neurons in each run of new images.
    #Spike vectors for excitatory neurons, and inhibitory neurons are initialized here
    SpikeExc = np.zeros((num_steps, N))
    SpikeInh = np.zeros((num_steps, N))
    
    #Converting the MNIST mage to Poisson spike trains
    te = k  # Image index
    image = X_test[te] / 4

    # max_rate = 100 #Original value
    max_rate = 1
    time_window = int(T_max * 1000)  
    dtM = dt * 1000  

    firing_rates = image * max_rate
    spike_trains = np.random.rand(28, 28, time_window) < (firing_rates[..., None] * dtM / 1000)
    SpikeIn = spike_trains.reshape(28 * 28, time_window).T  
    SpikeIn = SpikeIn.astype(int)
    SpikeIn_none = np.zeros((int(T_none / dt), I))  #Create zero input for T_none time steps
    SpikeIn = np.vstack((SpikeIn, SpikeIn_none)).astype(bool)  #Concatenate the zero input period with the active input period  
    
    for i in range(1, num_steps): #Perform the loop calculation for the network on each image

        #Synaptic conductance calculations for both excitatory and inhibitory synapses
        ge[i] = ge[i-1] * np.exp(-dt / tauge) + wExc @ SpikeIn[i]
        gi[i] = gi[i-1] * np.exp(-dt / taugi) + wIE @ SpikeInh[i-1]

        #Membrane potential calculations 
        Vexc[i] = Vexc[i-1] + dt * ((Erest - Vexc[i-1]) + ge[i] * (Eexc - Vexc[i-1]) + gi[i] * (Einh - Vexc[i-1])) / taue
        Vexc[i] = np.maximum(Vexc[i], -70 * (1e-3))
        
        #Excitatory Neuron Spike Detection & Reset
        spiking_neurons = Vexc[i] >= Vth_exc[i-1]
        SpikeExc[i, spiking_neurons] = 1
        Vexc[i, spiking_neurons] = Erest

        #When an excitatory neuron spikes, the inhibitory neuron its connected to fires.
        SpikeInh = SpikeExc

        Vth_exc[i] = Vth_base

    #Check for excitatory spikes. This gives us the number of spikes for each excitatory neuron.
    spike_counts = np.sum(SpikeExc, axis=0)
    #Accumulate spike counts for each class
    for l in range(O):  #Iterate over all possible digit classes (0-9)
        for m in range(N):  #Iterate over all excitatory neurons
            if neuron_class[m] == l:  #If neuron is assigned to this class
                response_class[l] += spike_counts[m]  #Accumulate spike counts

    #Compute average firing rate per class
    for l in range(O):
        count = np.sum(neuron_class == l)  #Count how many neurons belong to class l
        if count > 0:
            response_class[l] /= count  #Normalize by number of neurons in that class
    

    #Predict class based on highest response
    classification = np.argmax(response_class)
#     print(f"Predicted: {classification}, Actual: {label_test}")

    #Track accuracy
    total_predictions += 1  #Increment count of predictions made
    if classification == label_test:
        correct_predictions += 1

    #Check how well a random number generator does to check if network is no better than a random number generator
    num = np.random.randint(0, 10)  
    if num == label_test:
        correct_predictions2 += 1  



#Debugging: Print correct predictions and total predictions
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")

#Compute final accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f"Network Classification Accuracy: {accuracy:.2f}%")


#Compute final accuracy
accuracy2 = (correct_predictions2 / total_predictions) * 100
print(f"Random Number Generator Classification Accuracy: {accuracy2:.2f}%")
    
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f"Execution Time: {elapsed_time}")


#We plot the membrane potential for one of the excitatory neurons to observe behavior
plt.figure(figsize=(8, 4))
plt.plot(Vexc[:, 50] * 1000, label="Excitatory Neuron 50")  
plt.xlabel("Time Step")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.title("Membrane Potential Over Time")
plt.show()


#We plot the inhibitory and excitatory neurons from the last image run
#Check for excitatory spikes
spike_counts = np.sum(SpikeExc, axis=0)
print(f"Total spikes per neuron (Excitatory): {spike_counts}")
# #Check for inhibitory spikes
# spike_counts_inh = np.sum(SpikeInh, axis=0)
# print(f"Total spikes per neuron (Inhibitory): {spike_counts_inh}")

#Create a single figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

#Plot excitatory spike raster
axes[0].set_title("Excitatory Neurons - Spike Raster Plot")
for neuron_id in range(N):
    spike_times = np.where(SpikeExc[:, neuron_id] == 1)[0]
    axes[0].scatter(spike_times, np.full_like(spike_times, neuron_id), s=1, color="black") #Black for excitatory

axes[0].set_ylabel("Excitatory Neuron Index")

#Plot inhibitory spike raster
axes[1].set_title("Inhibitory Neurons - Spike Raster Plot")
for neuron_id in range(N):
    spike_times_inh = np.where(SpikeInh[:, neuron_id] == 1)[0]
    axes[1].scatter(spike_times_inh, np.full_like(spike_times_inh, neuron_id), s=1, color="red")  #Red for inhibitory

axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Inhibitory Neuron Index")

#Adjust layout for better spacing
plt.tight_layout()

#Show the combined plot
plt.show()






