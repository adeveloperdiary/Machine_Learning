import matplotlib.pyplot as plt


def showHistogram(data, bins=20):
    data.hist(bins=bins)
    plt.show()
