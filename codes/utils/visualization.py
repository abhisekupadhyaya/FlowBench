import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ldc_like(y, y_hat, idx, plot_path):
    plot_dir = os.path.dirname(plot_path)
    os.makedirs(plot_dir, exist_ok=True)
        
    #Create the 3x3 grid of plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    #Plot the ground truth
    im0 = axs[0, 0].imshow(y[idx][0], cmap='jet')
    axs[0, 0].set_title('u')
    fig.colorbar(im0, ax=axs[0, 0])
    im1  = axs[0, 1].imshow(y[idx][1], cmap='jet')
    axs[0, 1].set_title('v')
    fig.colorbar(im1, ax=axs[0, 1])
    im2 = axs[0, 2].imshow(y[idx][2], cmap='jet')
    axs[0, 2].set_title('p')
    fig.colorbar(im2, ax=axs[0, 2])
        
    #Plot the prediction
    im3 = axs[1, 0].imshow(y_hat[idx][0], cmap='jet')
    axs[1, 0].set_title('u')
    fig.colorbar(im3, ax=axs[1, 0])
    im4 = axs[1, 1].imshow(y_hat[idx][1], cmap='jet')
    axs[1, 1].set_title('v')
    fig.colorbar(im4, ax=axs[1, 1])
    im5 = axs[1, 2].imshow(y_hat[idx][2], cmap='jet')
    axs[1, 2].set_title('p')
    fig.colorbar(im5, ax=axs[1, 2])
        
    #Plot the error
    im6 = axs[2, 0].imshow(np.abs(y[idx][0] - y_hat[idx][0]), cmap='jet')
    axs[2, 0].set_title('u error')
    fig.colorbar(im6, ax=axs[2, 0])
    im7 = axs[2, 1].imshow(np.abs(y[idx][1] - y_hat[idx][1]), cmap='jet')
    axs[2, 1].set_title('v error')
    fig.colorbar(im7, ax=axs[2, 1])
    im8 = axs[2, 2].imshow(np.abs(y[idx][2] - y_hat[idx][2]), cmap='jet')
    axs[2, 2].set_title('p error')
    fig.colorbar(im8, ax=axs[2, 2])
        
    #Set the title for the plot
    fig.suptitle('Ground Truth, Prediction and Error')
        
    #Save the plot
    plt.savefig(plot_path)
        
    #Close the plot
    plt.close()