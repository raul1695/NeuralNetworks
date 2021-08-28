"""

Implementation of the simplest Neural Network, a single layer and neuron NN. 
The driver program uses pyplot to help the user visualize the classifier line transformation
through the NN training.

"""
import numpy as np 
import matplotlib.pyplot as pyplot


def plot_model(points, labels, weights, classes,l_rate, legend_desc = "Desired label"):
	colors = ['ro', 'g^']
	plot_points = np.delete(points, 2, axis = 1)
	for x in range(len(classes)):
		xx = []
		yy = []
		for ind in range(len(plot_points)):
			p = plot_points[ind]
			if(labels[ind] == classes[x]):
				xx.append(p[0])
				yy.append(p[1])
		pyplot.plot(xx,yy,colors[x], label = legend_desc + ":" +str(classes[x]))
	x1 = np.linspace(-5,5,30)
	x2 = -(weights[0]*x1 + weights[2]) / (weights[1])
	pyplot.plot(x1, x2, '-r')
	pyplot.axis([-4,4,-4,4])
	pyplot.xlabel("X1")
	pyplot.ylabel("X2")
	pyplot.title("Classifier Line Graph. Learn Rate : "+ str(l_rate))
	pyplot.legend()
	pyplot.show()

class NeuralNetwork(object):
	def __init__(self,rate):
		np.random.seed(1)
		self.weight_matrix = 2 * np.random.random((3,1)) - 1
		self.learning_rate = rate

	#Hard Limit Function
	def hard_limit(self,inputs):
		outs = np.zeros(inputs.shape)
		outs[inputs > 0.5] = 1
		return outs

	#forward propagantion
	def forward_propagation(self,inputs):
		prop = np.dot(inputs, self.weight_matrix)
		return self.hard_limit(prop)

	def pred(self,inputs):
		prop = self.forward_propagation(inputs)
		predictions = np.int8(prop)
		return predictions


	def train(self, train_inputs, train_desired_labels, num_train_itr = 10):
		N = train_inputs.shape[0]
		for iteration in range(num_train_itr):
			for index in range(train_inputs.shape[0]):
				actual_prediction  = self.pred(train_inputs[index,:])
				if(actual_prediction != train_desired_labels[index]):
					output = self.forward_propagation(train_inputs[index,:])
					error = train_desired_labels[index] - output
					dw = self.learning_rate * error * train_inputs[index]
					self.weight_matrix[:,0]+= dw
					plot_model(train_inputs, train_desired_labels, self.weight_matrix, [0,1],self.learning_rate)
   


def main():
	features = np.array([[1,1],[1,0],[0,1],[-1,-1],[-1,0],[-1,1]])
	labels = np.array([1,1,0,0,0,0])
	bias = np.ones((features.shape[0],1))
	features = np.append(features, bias, axis = 1)

	print("Graphs Below are with Learning rate  = 1\n\n\n\n")
	nn = NeuralNetwork(1)
	nn.train(features, labels, 100)

  #here is the data we wish to predict
	data_to_predict = np.array([[2,0],[2,1],[0,0],[-2,0]])
	#we need to add the bias in order for it to work with out function
	bias1 = np.ones((data_to_predict.shape[0],1))
	data_to_predict = np.append(data_to_predict, bias1, axis = 1)
   
	results = nn.pred(data_to_predict)
	print("The Graph below shows the result of our predictions as well as the classifier line \n\n\n\n")
	plot_model(data_to_predict, results, nn.weight_matrix, [0,1],nn.learning_rate, legend_desc="Predicted")

 
if __name__== "__main__":
  main()
