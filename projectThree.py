import random
import numpy as np

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
    
#coverts a diagram to an example 
def convertDiagramToExample(diagram, isDangerous):
    exampleX = []
    exampleY = 1 if isDangerous else 0
    for r in range(20):
        for c in range(20):
            if diagram[r][c] == 0:
                exampleX.extend([0,0,0,0,1])
            elif diagram[r][c] == 1:
                exampleX.extend([1,0,0,0,0])
            elif diagram[r][c] == 2:
                exampleX.extend([0,1,0,0,0])
            elif diagram[r][c] == 3:
                exampleX.extend([0,0,1,0,0])
            elif diagram[r][c] == 4:
                exampleX.extend([0,0,0,1,0])
    
    return exampleX, exampleY

def createDataSet(numExamples):
    X = []
    Y = []
    for i in range(numExamples):
        diagram, isDangerous, wire_to_cut = createDiagram()
        # exampleX, exampleY = convertDiagramToExample(diagram, isDangerous)
        X.append(np.array(diagram).flatten())
        Y.append(1 if isDangerous else 0)
    
    return np.array(X), np.array(Y)

def sigmoid(z):
    return 1/(1 + np.exp(-z)) # z is the dot product between weights and x

def logisticRegression(X, Y, learning_rate=0.1, epochs=1000):
    weights = np.random.randn(np.shape(X)[1] + 1) # initializes initial weights to random numbers 
    X_bias = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

    for epoch in range(epochs):
        rand_ind = np.random.permutation(len(Y)) # randomly choose data (prevent bias)
        
        for i in rand_ind:
            x_val = X_bias[i] 
            y_val = Y[i]

            preds = sigmoid(np.dot(weights, x_val))
            error = preds - y_val # error is prediction - actual value

            weights -= learning_rate * error * x_val

    return weights

# def predict(X, weights):
#     z = np.dot(X,weights)
#     predictions = []
    
#     for i in sigmoid(z):
#         if i > 0.5:
#             predictions.append(1)
#         else:
#             predictions.append(0)
    
#     return predictions

x_train, y_train = createDataSet(500)
trained = logisticRegression(x_train, y_train)
print("Trained Weights:", trained)

X_train_bias = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
preds_train = sigmoid(np.dot(X_train_bias, trained))
print("y_train: ", y_train)
print("preds_train: ", preds_train)
training_loss = -np.mean(y_train * np.log(preds_train) + (1 - y_train) * np.log(1 - preds_train))
print("Training Loss: ", training_loss)

predictions = (preds_train >= 0.5).astype(int)
accuracy = np.mean(predictions == y_train) * 100
print("Training Accuracy: ", accuracy, "%")