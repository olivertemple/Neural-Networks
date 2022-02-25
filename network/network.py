import numpy as np
from load import load_data_wrapper
import matplotlib.pyplot as plt

#Get the data and split it into training, validation and test sets
training_data, validation_data, test_data = load_data_wrapper()
num_train = len(training_data)
num_test = len(test_data)
num_val = len(validation_data)
print(num_train)
print(num_test)
X_train = []
y_train = []
for i in range(len(training_data)):
    X_train.append(training_data[i][0])
    y_train.append(training_data[i][1])
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []
for i in range(len(test_data)):
    X_test.append(test_data[i][0])
    y_test.append(test_data[i][1])
X_test = np.array(X_test)
y_test = np.array(y_test)

X_val = []
y_val = []
for i in range(len(validation_data)):
    X_val.append(validation_data[i][0])
    y_val.append(validation_data[i][1])
X_val = np.array(X_val)
y_val = np.array(y_val)

#define out activation function and its derivative
def sigmoid (x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

#define our loss function
def loss(predicted_output,desired_output):
    return 1/2*(desired_output-predicted_output)**2

#define out neural network
class NeuralNetwork() :
    def __init__ (self, inputLayerNeuronsNumber , hiddenLayerNeuronsNumber, outputLayerNeuronsNumber, learning_rate=0.1):
        #define the structure of our neural network
        self.inputLayerNeuronsNumber = inputLayerNeuronsNumber
        self.hiddenLayerNeuronsNumber = hiddenLayerNeuronsNumber
        self.outputLayerNeuronsNumber = outputLayerNeuronsNumber
        self.learning_rate = learning_rate

        #Randomly initialize the weights and biases for all layers
        self.hidden_weights = np.random.randn(hiddenLayerNeuronsNumber,inputLayerNeuronsNumber)*np.sqrt(2/inputLayerNeuronsNumber)
        self.hidden_bias = np.zeros([hiddenLayerNeuronsNumber,1])
        self.output_weights = np.random.randn(outputLayerNeuronsNumber,hiddenLayerNeuronsNumber)
        self.output_bias = np.zeros([outputLayerNeuronsNumber,1])
        self.loss = []
        
        
    def train(self, inputs, desired_output):
        #forward pass
        hidden_layer_in = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        hidden_layer_out = sigmoid(hidden_layer_in)
        
        output_layer_in = np.dot(self.output_weights, hidden_layer_out) + self.output_bias
        predicted_output = sigmoid(output_layer_in)
        
        #backward pass
        error = desired_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        
        error_hidden_layer = d_predicted_output.T.dot(self.output_weights)
        d_hidden_layer = error_hidden_layer.T * sigmoid_derivative(hidden_layer_out)
                
        self.output_weights += hidden_layer_out.dot(d_predicted_output.T).T * self.learning_rate
        self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
        
        self.hidden_weights += inputs.dot(d_hidden_layer.T).T * self.learning_rate
        self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate
        self.loss.append(loss(predicted_output,desired_output))
        
        
    def predict(self, inputs):
        #forward pass
        hidden_layer_in = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        hidden_layer_out = sigmoid(hidden_layer_in)
        output_layer_in = np.dot(self.output_weights, hidden_layer_out) + self.output_bias
        predicted_output = sigmoid(output_layer_in)
        return predicted_output

#create our neural network
nn=NeuralNetwork(784,350,10)

#train our neural network
n_epochs = 15
losses = []
mini_batches = int(num_train/n_epochs)
for i in range(X_train.shape[0]):#iterate through the training data
    inputs = np.array(X_train[i, :].reshape(-1,1))
    desired_output = np.array(y_train[i, :].reshape(-1,1))
    nn.train(inputs, desired_output)
    #print the loss every epoch
    if (i == 0):
        print("Epoch: 0, loss: {}".format(np.mean(nn.loss[0:mini_batches])))
        losses.append(np.mean(nn.loss[0:mini_batches]))
    if (i+1)%mini_batches == 0:
        print("Epoch: {}, loss: {}".format(int((i+1)/mini_batches),np.mean(nn.loss[i+1-mini_batches:i])))
        losses.append(np.mean(nn.loss[i+1-mini_batches:i]))

#calculate test accuracy
prediction_list = []
for i in range(X_test.shape[0]):
    inputs = np.array(X_test[i].reshape(-1,1))
    prediction_list.append(nn.predict(inputs))

correct_counter = 0
for i in range(len(prediction_list)):
    out_index = np.where(prediction_list[i] == np.amax(prediction_list[i]))[0][0]
    if y_test[i]==out_index:
        correct_counter+=1
accuracy = correct_counter/num_test
print("Test accuracy is : ",accuracy*100," %")


#calculate validation accuracy
val_prediction_list = []
for i in range(X_val.shape[0]):
    inputs = np.array(X_val[i].reshape(-1,1))
    val_prediction_list.append(nn.predict(inputs))

correct_counter = 0
for i in range(len(val_prediction_list)):
    out_index = np.where(val_prediction_list[i] == np.amax(val_prediction_list[i]))[0][0]
    if y_val[i]==out_index:
        correct_counter+=1
accuracy = correct_counter/num_val
print("Validation accuracy is : ",accuracy*100," %")

#plot loss
plt.plot(losses)
plt.style.use('seaborn')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#plot a small sample to give an example of the neural network
figure = plt.figure()
total_correct = 0
total_samples = 0
for i in range(1,10):
    index = np.random.randint(0,len(X_test))
    inputs = np.array(X_test[index].reshape(-1,1))
    prediction = np.argmax(nn.predict(inputs))
    label = y_test[index]
    plt.subplot(6, 10, i)
    plt.axis('off')
    plt.imshow(X_test[index].reshape(28,28), cmap='gray_r')
    print("Prediction from network: {}, label: {}".format(prediction , label))
    total_samples += 1
    if prediction == label:
        total_correct += 1
plt.show()
print("Accuracy from network: {}%".format(int(total_correct*100/total_samples)))

example = X_test[0]
example_label = y_test[0]
example_prediction = nn.predict(np.array(example).reshape(-1,1))
print(example)
print(example_label)
print(example_prediction)
print(np.argmax(example_prediction))
plt.axis('off')
plt.imshow(example.reshape(28,28), cmap='gray_r')
plt.show()