from matplotlib import pyplot as plt

def plot(exp, fig, axes): 
    axes[0].clear()
    axes[1].clear()
    axes[0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)], label="training loss")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].legend(loc="upper right")
    axes[1].plot([exp.history[k][1]['loss'] for k in range(exp.epoch)], label="evaluation loss")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Evaluation Loss')
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    fig.canvas.draw()