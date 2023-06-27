import math

import numpy as np

import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

skeleton = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_skeleton(
    save_path,
    joints,
    title,
    figsize=(5, 5),
    fps=20,
    radius=3,
    kinematic_tree=skeleton,
):
    matplotlib.use("Agg")

    title = "\n".join(textwrap.wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        fig.suptitle(title, fontsize=10)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = plt.axes(projection="3d")
    MINS = joints.min(axis=0).min(axis=0)
    MAXS = joints.max(axis=0).max(axis=0)
    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    colors = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#DD5A37",
        "#D69E00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
    ]

    frame_number = joints.shape[0]
    #     print(jointsset.shape)

    height_offset = MINS[1]
    joints[:, :, 1] -= height_offset
    joints[..., 0] -= joints[:, 0:1, 0]
    joints[..., 2] -= joints[:, 0:1, 2]

    trajec = joints[:, 0, [0, 2]]

    #     print(trajec.shape)

    def update(index):
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        ax.clear()

        init()

        """
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        """

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0

            ax.plot3D(
                joints[index, chain, 0],
                joints[index, chain, 1],
                joints[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    ani.save(save_path, fps=fps, dpi=100)

    plt.close()
