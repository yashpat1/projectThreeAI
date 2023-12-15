import random
import numpy as np
import matplotlib.pyplot as plt
import math

# Yash Patel, Siddh Parmar 
# Each group member contributed equally to the project 

# create the diagrams, which are 20x20 pixel images
def createDiagram():
    diagram = [[0 for _ in range(20)] for _ in range(20)]
    colors = [1, 2, 3, 4] # 1 = Red | 2 = Blue | 3 = Yellow | 4 = Green
    isDangerous = False
    wire_to_cut = 0

    row_first = random.random() < 0.5 # decides whether row or column is chosen first (50% chance)
    rowSelected = []
    columnSelected = []

    if row_first:
        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        randRow = random.randrange(1,21)
        rowSelected.append(randRow)
        for i in range(20):
            diagram[randRow - 1][i] = randColor

        if 3 in colors and 1 not in colors: # checks if Red wire is laid before Yellow wire to determine that image is Dangerous
            isDangerous = True

        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        randColumn = random.randrange(1,21)
        columnSelected.append(randColumn)
        for i in range(20):
            diagram[i][randColumn - 1] = randColor

        if 3 in colors and 1 not in colors:
            isDangerous = True
        

        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        wire_to_cut = randColor # signifies which wire to cut (3rd one laid down) if it is dangerous
        while True:
            randRow = random.randrange(1,21)
            if randRow not in rowSelected:
                break
        for i in range(20):
            diagram[randRow - 1][i] = randColor
        
        if 3 in colors and 1 not in colors:
            isDangerous = True


        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        while True:
            randColumn = random.randrange(1,21)
            if randColumn not in columnSelected:
                break
        for i in range(20):
            diagram[i][randColumn - 1] = randColor
    else:
        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        randColumn = random.randrange(1,21)
        columnSelected.append(randColumn)
        for i in range(20):
            diagram[i][randColumn - 1] = randColor

        if 3 in colors and 1 not in colors:
            isDangerous = True

        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        randRow = random.randrange(1,21)
        rowSelected.append(randRow)
        for i in range(20):
            diagram[randRow - 1][i] = randColor
        
        if 3 in colors and 1 not in colors:
            isDangerous = True

        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        wire_to_cut = randColor # signifies which wire to cut (3rd one laid down) if it is dangerous
        while True:
            randColumn = random.randrange(1,21)
            if randColumn not in columnSelected:
                break
        for i in range(20):
            diagram[i][randColumn - 1] = randColor
        
        if 3 in colors and 1 not in colors:
            isDangerous = True


        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        while True:
            randRow = random.randrange(1,21)
            if randRow not in rowSelected:
                break
        for i in range(20):
            diagram[randRow - 1][i] = randColor
    
    if isDangerous:
        return diagram, isDangerous, wire_to_cut 
    else: 
        return diagram, isDangerous, 0
    
# create a dangerous diagram, in which the red wire is always laid before the yellow wire
def createDangerousDiagram():
    diagram = [[0 for _ in range(20)] for _ in range(20)]
    colors = [1, 2, 4] # 1 = Red | 2 = Blue | 4 = Green (Yellow is initially removed)
    yellowWireAdded = False
    isDangerous = True
    wire_to_cut = 0


    row_first = random.random() < 0.5 # decides whether row or column is chosen first (50% chance)
    rowSelected = []
    columnSelected = []


    if row_first:
        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        randRow = random.randrange(1,21)
        rowSelected.append(randRow)
        for i in range(20):
            diagram[randRow - 1][i] = randColor


        if 1 not in colors: # checks if Red wire is laid before Yellow wire to determine that image is Dangerous
            yellowWireAdded = True
            colors.append(3) #Yellow wire can be selected after Red wire is laid
            isDangerous = True


        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        randColumn = random.randrange(1,21)
        columnSelected.append(randColumn)
        for i in range(20):
            diagram[i][randColumn - 1] = randColor


        if 1 not in colors and not yellowWireAdded: #make sure yellow wire has not been laid already
            yellowWireAdded = True
            colors.append(3)
            isDangerous = True
       
        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        wire_to_cut = randColor # signifies which wire to cut (3rd one laid down) if it is dangerous
        while True:
            randRow = random.randrange(1,21)
            if randRow not in rowSelected:
                break
        for i in range(20):
            diagram[randRow - 1][i] = randColor
       
        if 1 not in colors and not yellowWireAdded:
            yellowWireAdded = True
            colors.append(3)
            isDangerous = True


        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        while True:
            randColumn = random.randrange(1,21)
            if randColumn not in columnSelected:
                break
        for i in range(20):
            diagram[i][randColumn - 1] = randColor
       
        if 1 not in colors and yellowWireAdded:
            isDangerous = True
       
    else:
        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        randColumn = random.randrange(1,21)
        columnSelected.append(randColumn)
        for i in range(20):
            diagram[i][randColumn - 1] = randColor


        if 1 not in colors:
            yellowWireAdded = True
            colors.append(3)
            isDangerous = True
        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        randRow = random.randrange(1,21)
        rowSelected.append(randRow)
        for i in range(20):
            diagram[randRow - 1][i] = randColor
       
        if 1 not in colors and not yellowWireAdded:
            yellowWireAdded = True
            colors.append(3)
            isDangerous = True


        # column
        randColor = random.choice(colors)
        colors.remove(randColor)
        wire_to_cut = randColor # signifies which wire to cut (3rd one laid down) if it is dangerous
        while True:
            randColumn = random.randrange(1,21)
            if randColumn not in columnSelected:
                break
        for i in range(20):
            diagram[i][randColumn - 1] = randColor
       
        if 1 not in colors and not yellowWireAdded:
            yellowWireAdded = True
            colors.append(3)

        # row
        randColor = random.choice(colors)
        colors.remove(randColor)
        while True:
            randRow = random.randrange(1,21)
            if randRow not in rowSelected:
                break
        for i in range(20):
            diagram[randRow - 1][i] = randColor
       
        if 1 not in colors and yellowWireAdded:
            isDangerous = True
   
    if isDangerous:
        return diagram, isDangerous, wire_to_cut
    else:
        return diagram, isDangerous, 0
    
 
# converts a diagram to an example by implementing one hot encoding 4 digits based on the color of each pixel
def convertDiagramToExample(diagram, isDangerous):
    exampleX = []
    exampleY = 1 if isDangerous else 0
    for r in range(20): #coverts 400 pixel diagram to 1600 values
        for c in range(20):
            if diagram[r][c] == 1:
                exampleX.extend([1,0,0,0])
            elif diagram[r][c] == 2:
                exampleX.extend([0,1,0,0])
            elif diagram[r][c] == 3:
                exampleX.extend([0,0,1,0])
            elif diagram[r][c] == 4:
                exampleX.extend([0,0,0,1])
            else:
                exampleX.extend([0,0,0,0])
    
    return exampleX, exampleY

# creates a data set that contains a certain number of images (based on parameter)
def createDataSet(numExamples):
    X = []
    Y = []
    for i in range(numExamples):
        diagram, isDangerous, wire_to_cut = createDiagram()
        rotatedDiagram = diagram
        for i in range(4): #implements data augmentation by rotating diagram 3 times 
            exampleX, exampleY = convertDiagramToExample(rotatedDiagram, isDangerous)
            X.append(np.array(exampleX))
            Y.append(exampleY)
            rotatedDiagram = np.rot90(rotatedDiagram)
    
    return np.array(X), np.array(Y)

# sigmoid function for stochastic logistic regression
def sigmoid(z):
    return 1/(1 + np.exp(-z)) # z is the dot product between weights and x

# logistic regression algorithm that is used to make a decision on whether a diagram is dangerous or not
def logisticRegression(X, Y, learning_rate=0.005, epochs=1000):  
    weights = np.random.normal(loc=0, scale=0.00002, size=X.shape[1] + 1) # initializes initial weights to numbers between -0.00002 - 0.00002
    loss = []
    X_bias = np.concatenate([np.ones((X.shape[0], 1)), X], 1) # added a bias feature 

    epsilon = 1e-8 

    for epoch in range(epochs):
        rand_ind = np.random.permutation(len(Y)) # randomly choose data (prevent bias)
        epoch_loss = 0

        for i in rand_ind:
            x_val = X_bias[i] 
            y_val = Y[i]

            preds = sigmoid(np.dot(weights, x_val))
            preds = np.clip(preds, epsilon, 1-epsilon) # to help prevent log(0) or log(1) when computing loss
            error = preds - y_val # error is prediction - actual value
            weights -= learning_rate * (x_val * error)

            ind_loss = -y_val * np.log(preds) - (1 - y_val) * np.log(1 - preds) # finds individual cross-entropy loss
            epoch_loss += ind_loss

        loss.append(epoch_loss / len(rand_ind))

    return weights, loss

def accuracy(y_train, preds_train):
    preds = (preds_train >= 0.5).astype(int) # Utilized 0.5 as the classification threshold such that anything above or equal to 0.5 is 1, else 0
    return np.mean(preds == y_train) * 100 

# runs the training and testing of the first task using Logistic Regression
def task1(numTraining, numTesting):
    # training loss and accuracy
    x_train, y_train = createDataSet(numTraining)
    trained, loss = logisticRegression(x_train, y_train)

    X_train_bias = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
    preds_train = sigmoid(np.dot(X_train_bias, trained))
    print("Training Accuracy 1: ", accuracy(y_train, preds_train), "%")

    # test accuracy
    x_test, y_test = createDataSet(numTesting)
    X_test_bias = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
    preds_test = sigmoid(np.dot(X_test_bias, trained))
    print("Test Accuracy 1: ", accuracy(y_test, preds_test), "%")

    # Create graph for the loss over time
    epochs = range(1, len(loss) +1)
    plt.figure(figsize = (8,5))
    plt.plot(epochs, loss, marker='o', linestyle='-')
    plt.title('Loss Over Time for Step 1')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.show()

# converts a dangerous diagram to input vectors X and Y using one hot encoding and the counts of the pairs of colors within each 2 X 2 subarray of the diagram 
def convertDangerousDiagramToExample(diagram, wireToCut):
    exampleX = []
    exampleY = wireToCut
    for r in range(20): # one hot encoding x1 to x1600
        for c in range(20):
            if diagram[r][c] == 1:
                exampleX.extend([1,0,0,0])
            elif diagram[r][c] == 2:
                exampleX.extend([0,1,0,0])
            elif diagram[r][c] == 3:
                exampleX.extend([0,0,1,0])
            elif diagram[r][c] == 4:
                exampleX.extend([0,0,0,1])
            else:
                exampleX.extend([0,0,0,0])

    for i in range(len(exampleX)): # normalizes the one hot encoded vector using the mean component value of 76/1600 and standard deviation component value of sqrt(0.0452438) sqrt(72.39/1600)
        exampleX[i] -= (76/1600)
        exampleX[i] /= math.sqrt(0.0452438)

    mappedDiagram = [] # converts the diagram to a 2d array of the encoded values 
    for r in range(20):
        for c in range(20):
            if diagram[r][c] == 1:
                mappedDiagram.append([1,0,0,0])
            elif diagram[r][c] == 2:
                mappedDiagram.append([0,1,0,0])
            elif diagram[r][c] == 3:
                mappedDiagram.append([0,0,1,0])
            elif diagram[r][c] == 4:
                mappedDiagram.append([0,0,0,1])
            else:
                mappedDiagram.append([0,0,0,0])
            
    # (R,B,G,Y) Keys with all zeroes and only 1 non-zero value represents the pairs with white
    # dictionary tracking the counts of all the possible pairs of colors between two pixels (5 choose 2) = 15 pairs
    pairs = {(2,0,0,0):0, (1,1,0,0):0, (1,0,1,0):0, (1,0,0,1):0, (1,0,0,0):0, (0,2,0,0):0, (0,1,1,0):0, (0,1,0,1):0, (0,1,0,0):0, (0,0,2,0):0, (0,0,1,1):0, (0,0,1,0):0, (0,0,0,2):0, (0,0,0,1):0, (0,0,0,0):0}
    
    for r in range(0,19, 2): #For each 2 X 2 subarray of the diagram, it adds the top left pixel encoding to top right, top right to bottom right, bottom right to bottom left, and bottom left to top left and increment the count for the resulting pair
         for c in range(0, 19, 2): #tracks the intersections between pixels 
             pairs[tuple(sum(i) for i in zip(mappedDiagram[r * len(diagram) + c], mappedDiagram[r * len(diagram) + c + 1]))] += 1
             pairs[tuple(sum(i) for i in zip(mappedDiagram[r * len(diagram) + c + 1], mappedDiagram[(r + 1) * len(diagram) + c + 1]))] += 1
             pairs[tuple(sum(i) for i in zip(mappedDiagram[(r + 1) * len(diagram) + c + 1], mappedDiagram[(r + 1) * len(diagram) + c]))] += 1
             pairs[tuple(sum(i) for i in zip(mappedDiagram[(r + 1) * len(diagram) + c], mappedDiagram[r * len(diagram) + c]))] += 1
    
    for key in pairs: #adds the counts for each pair as components x1601 to x1615
        exampleX.append(pairs[key]) 

    return exampleX, exampleY

# creates a dangerous set 
def createDangerousSet(numExamples):
    X = []
    Y = []
    for i in range(numExamples): 
        diagram, isDangerous, wire_to_cut = createDangerousDiagram()
        if isDangerous:
            rotatedDiagram = diagram
            for i in range(4): #implements data augmentation by rotating diagram 3 times 
                exampleX, exampleY = convertDangerousDiagramToExample(rotatedDiagram, wire_to_cut)
                X.append(np.array(exampleX))
                Y.append(exampleY)
                rotatedDiagram = np.rot90(rotatedDiagram)
            
    return np.array(X), np.array(Y)

# create one-hot-encodings of the 4 possible outputs of the wire color to cut (1, 2, 3, 4)
def oneHotY(y, numClasses):
    numLabels = len(y)
    y_hot = np.zeros((numLabels, numClasses))

    for i in range(numLabels):
        y_hot[i, y[i] - 1] = 1

    return y_hot

# softmax function for softmax regression
def softmax(dot_prod):
    exps = np.exp(dot_prod - np.max(dot_prod, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)

# softmax regression algorithm that is used to make a decision on which colored-wire to cut 
def softmaxRegression(X, Y, learning_rate=0.01, beta=0.01, epochs=1000):
    classes = 4
    samples,features = X.shape

    weights = np.random.normal(loc=0, scale=0.002, size=(classes, features + 1))

    loss = []

    X_normal = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # normalize the input diagrams for preprocessing
    X_bias = np.concatenate((np.ones((samples,1)), X_normal), axis=1) # added the bias parameter
    one_hot_Y = oneHotY(Y, classes) # one-hot-vectors for each wire color

    decay_rate = 0.97 # decay rate is used to help change the learning rate as the algorithm runs 
    for epoch in range(epochs):
        learning_rate *= decay_rate
        epoch_loss = 0
        gradient = np.zeros_like(weights)

        for i in range(samples):
            x_val = X_bias[i]
            y_val = one_hot_Y[i]


            probs = softmax(np.dot(weights, x_val))
            epoch_loss += -np.sum(np.log(probs) * y_val) # Negative log likelihood function used to help determine loss
            gradient += np.outer(probs - y_val, x_val)
        
        weights -= learning_rate * (gradient/samples + beta * np.sign(weights)) # update weights to adjust model's parameters during training


        epoch_loss /= samples
        loss.append(epoch_loss)

    return weights, loss
 
def accuracySoftmax(y, preds):
    preds = np.argmax(preds, axis=1) + 1 # takes the index of the element that has the highest probability of being the wire to cut (add 1 since index+1 gives the number that represents the colored-wire)
    return np.round(np.mean(preds == y) * 100, 2)

# runs the training and testing of the second task using Softmax Regression
def task2(numTraining, numTesting):
    # training loss and accuracy
    x2_train, y2_train = createDangerousSet(numTraining)
    trained2, loss2 = softmaxRegression(x2_train, y2_train)

    X2_train_bias = np.concatenate((np.ones((x2_train.shape[0], 1)), x2_train), axis=1)
    preds2_train = softmax(np.dot(X2_train_bias, trained2.T))
    print("Training Accuracy 2: ", accuracySoftmax(y2_train, preds2_train), "%")

    # testing accuracy
    x2_test, y2_test = createDangerousSet(numTesting)
    X2_test_bias = np.concatenate((np.ones((x2_test.shape[0], 1)), x2_test), axis=1)
    preds2_test = softmax(np.dot(X2_test_bias, trained2.T))
    print("Test Accuracy 2: ", accuracySoftmax(y2_test, preds2_test), "%")

    # Create graph for the loss over time
    epochs = range(1, len(loss2) +1)
    plt.figure(figsize = (8,5))
    plt.plot(epochs, loss2, marker='o', linestyle='-')
    plt.title('Loss Over Time for Step 2')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.show()

# Run each task
task1(5000, 500)
task2(5000, 500)
