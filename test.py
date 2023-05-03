"""
    Testing code for different neural network configurations.
    Adapted for Python 3.5.2

    Usage in shell:
        python3.5 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""

import mnist_loader
import network2
from PIL import Image, ImageChops
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data = list(training_data)

# Contains all the 10,000 images where each element contains a tuple<flat(image), number>
test_data = list(test_data)[873]

# Grabs the image file
buffer = Image.open("number.jpg").convert("L")
buffer = ImageChops.invert(buffer)

# Grabs all the values and normalize each pixel value
image = list(buffer.getdata())
image = np.array(image)
image = (image - np.min(image)) / (np.max(image) - np.min(image))

# Flatten the image pixel values
image = np.reshape(image, (784, 1))

model = network2.load("model.json")
output = model.feedforward(image)
predict = np.argmax(output)

print("The predicted digit for the inputted image is:", predict)
buffer.show()

'''
Trains the model using Stochastic Gradient Descent
'''
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0 ,evaluation_data = validation_data, monitor_evaluation_accuracy=True)
# net.save("model.json");