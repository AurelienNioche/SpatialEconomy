import os
import matplotlib.pyplot as plt


from model.data_structure import ResultPool


def _plot(data, file_name, title=""):

    param_str = \
        "x: {}, stride: {}, movement_area: {}, vision_area: {}, " \
        "alpha: {:.2f}, tau: {:.2f}, map_width: {}, map_height: {}" \
        .format(data.parameters.x0,
                data.parameters.stride,
                data.parameters.movement_area,
                data.parameters.vision_area,
                data.parameters.alpha,
                data.parameters.tau,
                data.parameters.map_width,
                data.parameters.map_height)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines = ax.plot(data.indirect_exchanges_proportions)
    ax.set_ylim(0., 1.)
    ax.set_xlabel("t")
    ax.set_ylabel("Indirect exchanges proportion")
    ax.text(0.005, 0.005, param_str, transform=fig.transFigure, fontsize='x-small', color='0.5')
    ax.legend(lines, ("type-0", "type-1", "type-2"))
    ax.set_title(title)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    fig.savefig(file_name)
    plt.close(fig)


def plot_indirect_exchanges(data):

    if isinstance(data, ResultPool):

        for i, single_d in enumerate(data.data):
            file_name = "figures/{}/ind/{}_indirect_{}.pdf".format(data.file_name, data.file_name, i)
            title = "{}: {}".format(data.file_name, i)
            _plot(data=single_d, title=title, file_name=file_name)

    else:
        file_name = "figures/{}/{}_indirect.pdf".format(data.file_name, data.file_name)
        _plot(data=data, file_name=file_name)
