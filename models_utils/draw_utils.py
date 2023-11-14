import matplotlib.pyplot as plt


def draw_scatter(model, data, labels, sq=120, save_state=True, save_path=''):
    pred = model(data[:, sq-1:sq, :]).to('cpu').detach().numpy()
    true = labels.to('cpu').detach().numpy()
    plt.scatter(true[:, 0], true[:, 1], label='true', c="red")
    plt.scatter(pred[:, 0], pred[:, 1], label='prediction', c="blue")
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    if save_state:
        plt.savefig(save_path)
    plt.show()

