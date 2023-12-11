import random
import numpy as np
import matplotlib.pyplot as plt

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
            if diagram[r][c] == 1:
                exampleX.extend([1,0,0,0,0])
            elif diagram[r][c] == 2:
                exampleX.extend([0,1,0,0,0])
            elif diagram[r][c] == 3:
                exampleX.extend([0,0,1,0,0])
            elif diagram[r][c] == 4:
                exampleX.extend([0,0,0,1,0])
            else:
                exampleX.extend([0,0,0,0,1])
    
    return exampleX, exampleY

def createDataSet(numExamples):
    X = []
    Y = []
    for i in range(numExamples):
        diagram, isDangerous, wire_to_cut = createDiagram()
        rotatedDiagram = diagram
        for i in range(4):
            exampleX, exampleY = convertDiagramToExample(rotatedDiagram, isDangerous)
            X.append(np.array(exampleX))
            Y.append(exampleY)
            rotatedDiagram = np.rot90(rotatedDiagram)
        #exampleX, exampleY = convertDiagramToExample(diagram, isDangerous)
        #X.append(np.array(exampleX))
        #Y.append(exampleY)
    print("Percent of dangerous diagrams " + str(Y.count(1)/len(Y)))
    return np.array(X), np.array(Y)

def sigmoid(z):
    return 1/(1 + np.exp(-z)) # z is the dot product between weights and x

# def training_loss(y_train, preds_train):
#     for i in range(len(preds_train)):
#         if preds_train[i] == 1:
#             preds_train[i] = 1 - 1e-8
#         elif preds_train[i] == 0:
#             preds_train[i] = 1e-8
#     return np.mean(-y_train * np.log(preds_train) + (1 - y_train) * np.log(1 - preds_train))

def accuracy(y_train, preds_train):
    preds = (preds_train >= 0.5).astype(int)
    return np.mean(preds == y_train) * 100 


def logisticRegression(X, Y, learning_rate=0.01, lambda_val=0.00001, epochs=100):  #lr = 0.01, w = 0.000002
    weights = np.full(X.shape[1] + 1, -0.000002) # initializes initial weights to -0.02
    loss = []
    X_bias = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

    epsilon = 1e-8

    for epoch in range(epochs):
        rand_ind = np.random.permutation(len(Y)) # randomly choose data (prevent bias)
        epoch_loss = 0
        
        for i in rand_ind:
            #l1_reg = lambda_val * np.sum(np.abs(weights))
            x_val = X_bias[i] 
            y_val = Y[i]

            preds = sigmoid(np.dot(weights, x_val))
            preds = np.clip(preds, epsilon, 1-epsilon)
            error = preds - y_val # error is prediction - actual value
            weights -= learning_rate * (x_val * error)

            ind_loss = -y_val * np.log(preds) - (1 - y_val) * np.log(1 - preds)
            epoch_loss += ind_loss
        

        loss.append(epoch_loss / len(rand_ind))

    return weights, loss

# training loss and accuracy
x_train, y_train = createDataSet(5000)
trained, loss = logisticRegression(x_train, y_train)

X_train_bias = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
preds_train = sigmoid(np.dot(X_train_bias, trained))
   
print("Training Accuracy: ", accuracy(y_train, preds_train), "%")

epochs = range(1, len(loss) +1)
plt.figure(figsize = (8,5))
plt.plot(epochs, loss, marker='o', linestyle='-')
plt.title('Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.grid(True)
plt.show()