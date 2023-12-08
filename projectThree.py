import random
import numpy as np

# create the diagrams, which are 20x20 pixel images
def createDiagram ():
    diagram = [[0 for _ in range(20)] for _ in range(20)]
    colors = [1, 2, 3, 4] # 1 = Red | 2 = Blue | 3 = Yellow | 4 = Green
    isDangerous
    wire_to_cut

    row_first = random.random() < 0.5 # decides whether row or column is chosen first (50% chance)
    rowSelected = []
    columnSelected = []

    if row_first:
        randColor = random.choice(colors)
        colors.remove(randColor)
        randRow = random.randrange(1,21)
        rowSelected.append(randRow)
        for i in range(20):
            diagram[randRow][i] = randColor

        if 3 in colors and 1 not in colors:
            isDangerous = True

        
        randColor = random.choice(colors)
        colors.remove(randColor)
        randColumn = random.randrange(1,21)
        columnSelected.append(randColumn)
        for i in range(20):
            diagram[i][randColumn] = randColor

        if 3 in colors and 1 not in colors:
            isDangerous = True
        

        randColor = random.choice(colors)
        colors.remove(randColor)
        wire_to_cut = randColor
        while True:
            randRow = random.randrange(1,21)
            if randRow not in rowSelected:
                break
        for i in range(20):
            diagram[randRow][i] = randColor
        
        if 3 in colors and 1 not in colors:
            isDangerous = True


        randColor = random.choice(colors)
        colors.remove(randColor)
        while True:
            randColumn = random.randrange(1,21)
            if randColumn not in columnSelected:
                break
        for i in range(20):
            diagram[i][randColumn] = randColor
    else:
        randColor = random.choice(colors)
        colors.remove(randColor)
        randColumn = random.randrange(1,21)
        columnSelected.append(randColumn)
        for i in range(20):
            diagram[i][randColumn] = randColor

        if 3 in colors and 1 not in colors:
            isDangerous = True

        
        randColor = random.choice(colors)
        colors.remove(randColor)
        randRow = random.randrange(1,21)
        rowSelected.append(randRow)
        for i in range(20):
            diagram[randRow][i] = randColor
        
        if 3 in colors and 1 not in colors:
            isDangerous = True

        
        randColor = random.choice(colors)
        colors.remove(randColor)
        wire_to_cut = randColor
        while True:
            randColumn = random.randrange(1,21)
            if randColumn not in columnSelected:
                break
        for i in range(20):
            diagram[i][randColumn] = randColor
        
        if 3 in colors and 1 not in colors:
            isDangerous = True


        randColor = random.choice(colors)
        colors.remove(randColor)
        while True:
            randRow = random.randrange(1,21)
            if randRow not in rowSelected:
                break
        for i in range(20):
            diagram[randRow][i] = randColor
    
    if isDangerous:
        return diagram, isDangerous, wire_to_cut
    else: 
        return diagram, isDangerous, 0