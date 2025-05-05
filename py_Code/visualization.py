import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
raw_data=pd.read_csv("kpmi_ru_data.csv")
types=[["e","i"],["s","n"],["t","f"],["j","p"]]
def visualization_3d(params):
    count=raw_data[params].value_counts()
    x_coord=[i[0] for i in count.index]
    y_coord=[j[1] for j in count.index]
    z_value=count.values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coord, y_coord, z_value, c=z_value, cmap='viridis', marker='o')
    ax.set_xlabel(f'Score of {params[0]}')
    ax.set_ylabel(f'Score of {params[1]}')
    ax.set_zlabel('Frequency')
    plt.title(f"Distribution of {params[0]} and {params[1]}")
    plt.show()
def visualization_2d(param):
    count=raw_data[param].value_counts()
    x_coord=count.index
    y_value=count.values
    plt.scatter(x_coord,y_value)
    plt.xlabel(f"Score of {param}")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {param}")
    plt.show()
for i in types:
    visualization_3d(i)
visualization_2d("e")