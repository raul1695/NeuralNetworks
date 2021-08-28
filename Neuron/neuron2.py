"""

Another implementation of a single layer, single neuron Neural Network (NN). 
Linear activation function, foward propagation.

"""

import numpy as np 
import matplotlib.pyplot as pyplot


def plot_func(points, labels, weights, classes,l_rate):
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
		pyplot.plot(xx,yy,colors[x], label = "Desired Label : "+str(classes[x]))
	x1 = np.linspace(-5,5,30)
	x2 = -(weights[0]*x1 + weights[2] / weights[1])
	pyplot.plot(x1, x2, '-r')
	pyplot.axis([-4,4,-4,4])
	pyplot.xlabel("X1")
	pyplot.ylabel("X2")
	pyplot.title("Classifier Line Graph. Learn Rate : "+ str(l_rate))
	pyplot.legend()
	pyplot.show()



def learning_graph(learning_rate, costs, epochs,itr):
	x = np.linspace(0,40,costs.shape[0])
	if(np.sum(costs,axis = 0) == 0):
		return
	pyplot.plot(x,costs,'-r')
	pyplot.title("Cost Function Learn Rate = "+ str(learning_rate)+ " Iteration #"+str(itr))
	pyplot.xlabel("Iterations")
	pyplot.ylabel("Average cost")
	pyplot.show()




class NeuralNetwork(object):
	def __init__(self,rate):
		np.random.seed(1)
		self.weight_matrix = 2 * np.random.random((3,1)) - 1
		self.learning_rate = rate
    
		#history variable
		self.history_var_w = []

		#history variable
		self.history_var_cost = []


	#forward propagantion
	def forward_propagation(self,inputs):
		outs = np.dot(inputs, self.weight_matrix)
		return outs


  #Training method
  # train_inputs (the X1, X2 inputs in array form)
  # train_desired_labels (the desired labels)
	def train_GDL(self, train_inputs, train_desired_labels, num_train_itr = 1000):
		N = train_inputs.shape[0]
		for iteration in range(num_train_itr):
			if((iteration + 1) % 5  == 0):
				c = [-1,1]
				print("\n\n\n\n")
				plot_func(train_inputs,train_desired_labels,self.weight_matrix,c,self.learning_rate)
				print("\n\n\n\n")

			actual_outputs = self.forward_propagation(train_inputs)			
			error = train_desired_labels - actual_outputs
			adjustment = (self.learning_rate/N) * np.sum(np.multiply(error,train_inputs), axis = 0)


			#adjust the weights
			self.weight_matrix[:,0] += adjustment

			#calculate cost for plotting later
			cost = float( ((1/(2*N) )) * np.sum(np.square(error),axis = 0) )
			self.history_var_cost.append(cost)
			self.history_var_w.append((self.weight_matrix[0],self.weight_matrix[1],self.weight_matrix[2]))
			learning_graph(self.learning_rate, np.asarray(self.history_var_cost), num_train_itr,iteration)

   


def main():
	features = np.array([[1,1],[1,0],[0,1],[-1,-1],[.5,3],[.7,2],[-1,0],[-1,1],[2,0],[-2,-1]])
	labels = np.array([1,1,-1,-1,1,1,-1,-1,1,-1])
	bias = np.ones((features.shape[0],1))
	features = np.append(features, bias, axis = 1)

	print("Graphs Below are with Learning rate  = .1\n\n\n\n")
	nn = NeuralNetwork(0.1)
	nn.train_GDL(features, np.expand_dims(labels, axis = 1), 50)


	print("Graphs Below are with Learning rate  = .5\n\n\n\n")
	nn = NeuralNetwork(0.5)
	nn.train_GDL(features, np.expand_dims(labels, axis = 1), 50)


	print("Graphs Below are with Learning rate  = 1\n\n\n\n")
	nn = NeuralNetwork(1)
	nn.train_GDL(features, np.expand_dims(labels, axis = 1), 50)

  
if __name__== "__main__":
  main()
