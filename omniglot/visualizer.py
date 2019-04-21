from matplotlib import pyplot as plt
import os

class visualizer():
    def __init__(exp_name, row, col):
        self.fig, self.axs = plt.subplots(row, col)
        self.row, self.col = row, col
        self.exp_name = exp_name
        os.makedir(exp_name, exist_ok=True)
    def draw_imgs(images):
        img_channel = images.shape[3]
        if img_channel == 1:
            cmap = 'gray'
        else:
            cmap = 'rgb'
        for j in range(self.row):
            for i in range(self.col):
                axs[j, i].imshow(images[j*self.col+i, :, :, img_channel], cmap = cmap)
                axs[j, i].axis('off')
    def save_fig(name):
        filename = "{}.png".format(name)
        path = os.path.join(self.exp_name, filename)
        self.fig.savefig(path)