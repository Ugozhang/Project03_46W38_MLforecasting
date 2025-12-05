import matplotlib.pyplot as plt

def single_var_plot(x, y, xlabel=None, ylabel=None, title=None):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()