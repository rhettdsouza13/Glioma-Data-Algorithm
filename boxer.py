import matplotlib.pyplot as plt

def customized_box_plot(percentiles, axes, redraw = True, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """
    n_box = len(percentiles)
    box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, *args, **kwargs)
    # Creates len(percentiles) no of box plots

    min_y, max_y = float('inf'), -float('inf')
    ax.set_xticklabels(['Artificial Neural Network', 'Support Vector Machine', 'Logistic Regression', 'Decision Tree'])
    ax.set_xticks([1.5, 4.5, 7.5, 10.5])

    for box_no, (q1_start,
                 q2_start,
                 q3_start,
                 q4_start,
                 q4_end,
                 fliers_xy) in enumerate(percentiles):

        # Lower cap
        box_plot['caps'][2*box_no].set_ydata([q1_start, q1_start])
        # xdata is determined by the width of the box plot

        # Lower whiskers
        box_plot['whiskers'][2*box_no].set_ydata([q1_start, q2_start])

        # Higher cap
        box_plot['caps'][2*box_no + 1].set_ydata([q4_end, q4_end])

        # Higher whiskers
        box_plot['whiskers'][2*box_no + 1].set_ydata([q4_start, q4_end])

        # Box
        box_plot['boxes'][box_no].set_ydata([q2_start,
                                             q2_start,
                                             q4_start,
                                             q4_start,
                                             q2_start])

        # Median
        box_plot['medians'][box_no].set_ydata([q3_start, q3_start])

        # Outliers
        # if fliers_xy is not None and len(fliers_xy[0]) != 0:
        #     # If outliers exist
        #     box_plot['fliers'][box_no].set(xdata = fliers_xy[0],
        #                                    ydata = fliers_xy[1])
        #
        #     min_y = min(q1_start, min_y, fliers_xy[1].min())
        #     max_y = max(q4_end, max_y, fliers_xy[1].max())


        min_y = min(q1_start, min_y)
        max_y = max(q4_end, max_y)

        # The y axis is rescaled to fit the new box plot completely with 10%
        # of the maximum value at both ends
        axes.set_ylim([0, 1.0])

    # If redraw is set to true, the canvas is updated.
    if redraw:
        ax.figure.canvas.draw()

    return box_plot

# 0.65845
# 0.7369425
# 0.82
# 0.8939275
# 0.97242

# percentiles = [[0.65845,0.7369425,0.82,0.8939275,0.97242, []], [0.22478,0.3635925,0.50,0.6412175,0.78003,[]],
# [0.62948,0.7112975,0.79312,0.8749325,0.95675,[]],
# ]
percentiles = []
st_dt_f = open("stat_dat.txt", 'r')
# print st_dt_f.readlines()[0].strip().split()
# print st_dt_f.readlines()[0]
st_dt = [x.strip().split() for x in st_dt_f.readlines()]
print st_dt
# st_dt = [float(y) for y in x for x in st_dt]
data = []
for x in st_dt:
    ys = []
    for y in x:
        ys.append(float(y))
    ys.append([])
    data.append(ys)

# st_dt = [x.append([]) for x in st_dt]
print data
fig, ax = plt.subplots()
b = customized_box_plot(data, ax, redraw=True, notch=0, sym='+', vert=1, whis=1.5)
plt.show()
