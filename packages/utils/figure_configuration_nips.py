import matplotlib


def figure_configuration_nips():

    # run this code before the plot command

    # Width: column width: 8.8 cm; page width: 18.1 cm.

    # width & height of the figure
    k_scaling = 0.393701 #cm to inches
    # scaling factor of the figure
    # (You need to plot a figure which has a width of (8.8 * k_scaling)
    # in MATLAB, so that when you paste it into your paper, the width will be
    # scalled down to 8.8 cm  which can guarantee a preferred clearness.

    k_width_height = 1.5#1.3  # width:height ratio of the figure

    fig_width = 13.968/2 * k_scaling
    fig_height = fig_width / k_width_height

    # ## figure margins
    # top = 0.5  # normalized top margin
    # bottom = 3	# normalized bottom margin
    # left = 4	# normalized left margin
    # right = 1.5  # normalized right margin
    fontsize=8
    params = {'axes.labelsize': fontsize,  # fontsize for x and y labels (was 10)
              'axes.titlesize': fontsize,
              'font.size': fontsize,  # was 10
              'legend.fontsize': fontsize,  # was 10
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'font.serif': ['Times New Roman'],
              'lines.linewidth': 2.5,
              'axes.linewidth': 1,
              'axes.grid': True,
              'savefig.format': 'pdf',
              'axes.xmargin': 0,
              'axes.ymargin': 0,
              'savefig.pad_inches': 0,
              'legend.markerscale': 1,
              'savefig.bbox': 'tight',
              'lines.markersize': 2,
              'legend.numpoints': 4,
              'legend.handlelength': 3.5,
              'text.usetex': True
              }

    matplotlib.rcParams.update(params)
