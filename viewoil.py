import matplotlib.pyplot as plt
import pickle
import sys

if len(sys.argv) > 1:
    pklFile = str(sys.argv[1])

    with open(pklFile, 'rb') as file:
        coords = pickle.load(file)

    X, Y, Z = coords.X, coords.Y, coords.Z

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=100, cmap="cool")
    plt.colorbar(label='Tangential Acceleration')
    plt.xlabel('Board')
    plt.ylabel('Feet')
    plt.title('Oil Distribution')
    plt.show()
else:
    print("Please specify file: viewoil.py <path to file>")