from matplotlib import pyplot as plt
from typing import List, Tuple, Dict

def extract(key: str, item: Dict[str, float]) -> float:
    value = item[key]
    return value.cpu().item() if hasattr(value, 'cpu') else value

def plot_results_panels(losses: List[Dict[str, float]], plot_configs: List[Tuple[str, str, str, str]], figsize: Tuple[int, int] = (16, 4), marking_color: str = 'purple') -> plt.Figure:
    fig, axs = plt.subplots(1, len(plot_configs), figsize=figsize, sharex=True)

    epochs = [item['Epoch'] for item in losses]

    for ax, (key, title, color, low_or_high) in zip(axs, plot_configs):
        values = [extract(key, item) for item in losses]
        ax.plot(epochs, values, label=key, color=color)
        
        if low_or_high == 'low':
            value = min(values)
            epoch = epochs[values.index(value)]
        elif low_or_high == 'high':
            value = max(values)
            epoch = epochs[values.index(value)]
        else:
            value = 0
            epoch = 0
        
        best_epoch_text = f"Best: {value:.2f} at Epoch {epoch}"
        ax.scatter(epoch, value, color=marking_color, zorder=5)
        ax.axvline(epoch, color=marking_color, linestyle='--', linewidth=1)
        
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend([key, best_epoch_text])
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
    return fig