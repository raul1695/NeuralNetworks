"""
The implementation of a complete NN using the Keras Library. Using python's pyplot for visualization.


"""

import numpy as np 
import matplotlib.pyplot as plt, matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense



def plot_func(points, labels = None, classes = None, graph = None, graph_title = None):
	colors = ['ro', 'g^']
	if(graph_title == None):
		graph_title = "Graph"
	if(graph == None):
		plot_points = points
		for x in range(len(classes)):
			xx = []
			yy = []
			for ind in range(len(plot_points)):
				p = plot_points[ind]
				if(labels[ind] == classes[x]):
					xx.append(p[0])
					yy.append(p[1])
			pyplot.plot(xx,yy,colors[x], label = "Desired Label : "+str(classes[x]))
		pyplot.axis([-4,4,-4,4])
		pyplot.xlabel("X1")
		pyplot.xlabel("X2")
		pyplot.title(graph_title)
		pyplot.legend()
	else:
		plot_points = points
		if(classes != None):
			for x in range(len(classes)):
				xx = []
				yy = []
				for ind in range(len(plot_points)):
					p = plot_points[ind]
					if(labels[ind] == classes[x]):
						xx.append(p[0])
						yy.append(p[1])
				graph.plot(xx,yy,colors[x], label = "Desired Label : "+str(classes[x]))
		graph.axis([-4,4,-4,4])
		graph.set_xlabel("X1")
		graph.set_ylabel("X2")
		graph.set_title(graph_title)
		graph.legend()

 
def plot_curve(data_points, graph, graph_title = "Title Here"):
	epoch = np.arange(np.asarray(data_points).shape[0])
	graph.plot(epoch,data_points)
	graph.set_xlabel("X1")
	graph.set_ylabel("X2")
	graph.set_title(graph_title)
 

def plot_classifier(weights,graph):
  x1 = np.linspace(-20,20,len(weights))
  x2 = ((weights[0] *x1)+weights[2])/weights[1]


  
  graph.plot(x1,x2, '-r')


"""

The function was obtained from the following source : https://www.datahubbs.com/deep-learning-101-first-neural-network-keras/

"""
def plot_decision_boundary(prediction_model, X, Y):
	x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
	y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
	spacing = min(x_max - x_min, y_max - y_min) / 100
	XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                   np.arange(y_min, y_max, spacing))
	data = np.hstack((XX.ravel().reshape(-1,1), 
                      YY.ravel().reshape(-1,1)))
	db_prob = prediction_model.predict(data)
	clf = np.where(db_prob<0.5,0,1)
	Z = clf.reshape(XX.shape)
	plt.figure(figsize=(10,8))
	plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:,0], X[:,1], c=Y, 
                cmap=plt.cm.Spectral)


 
def main():
	features = np.array([[0,0],[0,1],[1,0],[1,1]])
	labels = np.array([0,1,1,0])
	figure, axis = plt.subplots(2,2, figsize=(15,15))
 
 #Plotting the graph for part B of the question
	plot_func(features, labels = labels, classes = [0,1], graph = axis[0,0])
 
	model_m = Sequential([
    Dense(units = 2, input_dim = 2, activation = 'sigmoid'),
    Dense(units = 1, activation = 'sigmoid'),
])

	model_m.summary()
	model_m.compile(loss = "binary_crossentropy",optimizer = SGD(lr=0.1),metrics=['accuracy'])
	history = model_m.fit(features,labels, 
          batch_size= 1, 
          epochs = 200, 
          verbose = 2
         )







 
  
  #Ploting loss
	plot_curve(history.history['loss'], graph = axis[1,0], graph_title = "Loss for NN with 3 Nodes")


 
 #Second model with 2 extra nodes in the first layer and epoch set to 400


	model_mm = Sequential([
    Dense(units = 4, input_dim = 2, activation = 'sigmoid'),
    Dense(units = 1, activation = 'sigmoid'),
])
	model_mm.summary()
	model_mm.compile(loss = "binary_crossentropy",optimizer = SGD(lr=0.1),metrics=['accuracy'])
	history = model_mm.fit(features,labels, 
          batch_size= 1, 
          epochs = 400, 
          verbose = 2
         )
 
  #Ploting loss

	plot_curve(history.history['loss'], graph = axis[1,1], graph_title = "Loss for NN with 3+2 Nodes")
 
 
	results = model_mm.evaluate(x = features,y = labels, batch_size = 1)
	plot_decision_boundary(model_mm, features, labels)
	print(results)








 




  
if __name__== "__main__":
  main()
