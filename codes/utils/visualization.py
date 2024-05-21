import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_tensor(tensor_data, interval=100, cmap='viridis'):
    """
    Creates an animation of the slices of a tensor, typically used for visualizing model outputs.

    Args:
        tensor_data (Tensor): A 3D or 4D tensor to be visualized. If 4D, the function assumes (batch_size, channels, height, width).
        interval (int): Time interval between frames in milliseconds.
        cmap (str): Color map used for the animation.

    Returns:
        HTML: HTML object containing the animation.
    """
    # Ensure tensor is on CPU and convert to numpy
    tensor_data = tensor_data.cpu().numpy()

    # Handle 4D tensor by selecting the first element and first channel
    if tensor_data.ndim == 4:
        tensor_data = tensor_data[0, 0]

    fig, ax = plt.subplots()
    plt.close(fig)  # Close the figure to prevent it from displaying statically

    # Initial plot
    img = ax.imshow(tensor_data[:, :, 0], cmap=cmap)

    def update(frame):
        """
        Update function for animation, changes the displayed slice.

        Args:
            frame (int): Frame index to display.
        """
        img.set_data(tensor_data[:, :, frame])
        ax.set_title(f"Frame {frame + 1}")
        return img,

    ani = FuncAnimation(fig, update, frames=tensor_data.shape[2], interval=interval, blit=True)

    return HTML(ani.to_html5_video())

def plot_tensor(tensor_data, index=0, channel=0, cmap='viridis'):
    """
    Plots a single slice of a tensor.

    Args:
        tensor_data (Tensor): A 3D or 4D tensor to be plotted.
        index (int): Index of the batch in case of a 4D tensor.
        channel (int): Channel index to plot in case of a 4D tensor.
        cmap (str): Color map used for the plot.
    """
    # Ensure tensor is on CPU and convert to numpy
    tensor_data = tensor_data.cpu().numpy()

    # Handle 4D tensor by selecting specified index and channel
    if tensor_data.ndim == 4:
        tensor_data = tensor_data[index, channel]

    plt.imshow(tensor_data, cmap=cmap)
    plt.colorbar()
    plt.title("Tensor Slice Visualization")
    plt.show()