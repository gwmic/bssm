import numpy as np
from pykrige.ok import OrdinaryKriging
import pickle
import matplotlib.pyplot as plt
import cv2


def secondDerivative(poly):
    # Compute the second derivative of the polynomial
    return np.poly1d(np.polyder(poly, 2))


def generatePoints(poly, xMin, xMax, divisons):
    # Evaluate the polynomial and its second derivative
    yValues = np.linspace(xMin, xMax, divisons)
    xValues = poly(yValues)
    zValues = abs(secondDerivative(poly)(yValues))
    yValues = np.append(yValues, yValues)
    xValues = np.append(xValues, (39 - xValues))
    zValues = np.append(zValues, zValues)
    return xValues, (yValues/11.1833), (zValues*1000)


class Coords:
    def __init__(self, X, Y, Z):
        self.X = self.Y = self.Z = X, Y, Z


def heatMap(data):
    xArr, yArr, zArr = np.array([])

    for shot in data.shotArr:
        xPoly, yPoly, zPoly = generatePoints(shot.polyReal, 0, 671, 100)
        xArr = np.append(xArr, xPoly)
        yArr = np.append(yArr, yPoly)
        zArr = np.append(zArr, zPoly)

    # Create a grid of points to interpolate onto
    xi = np.linspace(0, 39, 250)
    yi = np.linspace(0, 60, 250)
    X, Y = np.meshgrid(xi, yi)

    # Set up the Ordinary Kriging model
    OKmodel = OrdinaryKriging(xArr, yArr, zArr, variogram_model='gaussian')

    # Predict on the grid
    Z, _ = OKmodel.execute('grid', xi, yi)

    coords = Coords(X, Y, Z)
    with open('oildistribution.pkl', 'wb') as file:
        pickle.dump(coords, file)
    
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=100, cmap="cool")
    plt.colorbar(label='Tangential Acceleration')
    plt.xlabel('Board')
    plt.ylabel('Feet')
    plt.title('Oil Distribution')
    plt.savefig('.heatmap.png')
    image = cv2.imread('.heatmap.png')
    data.window["Heatmap"] = image