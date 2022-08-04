import plotly.graph_objects as go
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import open3d as o3d
from summer2022_toolbox.visualization_3D_objects import *
from summer2022_toolbox.model_PCA import *

def get_all_MSE(X_test_flat, V, X_avg, n_features):
    MSE_all = []
    for i_fet in range(n_features):
        total_mse = 0
        features = V[:, :i_fet]
        for test_flat in X_test_flat:
            weights = (test_flat - X_avg.flatten('F'))@features
            fitted_model = X_avg.flatten('F') + features @ weights
            mse = ((fitted_model - test_flat)**2).mean()
            total_mse += mse
        MSE_all.append(total_mse/(len(X_test_flat)))
    return MSE_all

def plot_MSE_n_feature(X_test_flat, V, X_avg, n_features):
    MSE_all = get_all_MSE(X_test_flat, V, X_avg, n_features)

    plt.clf
    plt.plot(range(n_features), MSE_all, '.-')
    plt.axis('tight')
    plt.xlabel('Number of Features')
    plt.ylabel('MSE')
    plt.show()


# Create figure
def draw3DpointsSlider(features, mod_place, X_avg, weights_all):
    """
    features : np.ndarray (3n features, number of total chosen eigenvalues)
    mod_weights : np.ndarray (number of total chosen eigenvalues, 1), while keeping the chosen eigenvalue unchanged,
                  the others set to 0.
    X_avg : np.ndarray (3, n) vertices of the mean shape
    """
    fig = go.Figure()
    # Add traces, one for each slider step

    for step in np.arange(0, 1.1, 0.1):
        mod_weights = np.zeros(len(features[0]))
        normalize = np.max(weights_all[mod_place]) - np.min(weights_all[mod_place])
        mod_weights[mod_place] = step * normalize + np.min(weights_all[mod_place])

        new_shape = np.add(X_avg, np.matmul(features, (mod_weights)).reshape(-1, 3).T)

        fig.add_trace(
            go.Scatter3d(
                visible=False,
                name="𝜈 = " + str(step),
                x=new_shape[0],
                y=new_shape[1],
                z=new_shape[2],
                mode='markers', marker=dict(
                    size=1,
                    color=new_shape[2],  # set color to an array/list of desired values
                    colorscale='Viridis'
                )
            )

        )

    # Make 10th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []

    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Feature No.%d" % (mod_place)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=50,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[np.min(X_avg[0]) - 100, np.max(X_avg[0]) + 100], ),
            yaxis=dict(range=[np.min(X_avg[1]) - 100, np.max(X_avg[1]) + 100], ),
            zaxis=dict(range=[np.min(X_avg[2]) - 100, np.max(X_avg[2]) + 100], ),
            aspectratio=dict(x=((np.max(X_avg[0]) + 100) - (np.min(X_avg[0]) - 100)) / 1000,
                             y=((np.max(X_avg[1]) + 100) - (np.min(X_avg[1]) - 100)) / 1000,
                             z=((np.max(X_avg[2]) + 100) - (np.min(X_avg[2]) - 100)) / 1000)
        ),
        width=1000,
        height=700,

        sliders=sliders

    )

    fig.show()


# Create figure
def draw3DpointsSliderRecon(features, mod_place, X_avg, weights_all):
    """
    features : np.ndarray (3n features, number of total chosen eigenvalues)
    mod_weights : np.ndarray (number of total chosen eigenvalues, 1), while keeping the chosen eigenvalue unchanged,
                  the others set to 0.
    X_avg : np.ndarray (3, n) vertices of the mean shape
    """
    fig = go.Figure()
    # Add traces, one for each slider step

    for step in np.arange(0, 1.1, 0.1):
        mod_weights = np.zeros(len(features[0]))
        normalize = np.max(weights_all[mod_place]) - np.min(weights_all[mod_place])
        mod_weights[mod_place] = step * normalize + np.min(weights_all[mod_place])

        new_shape = np.add(X_avg, np.matmul(features, (mod_weights)).reshape(-1, 3).T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_shape.T)

        # # visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 100)
        X, F = np.asarray(mesh.vertices).T, np.asarray(mesh.triangles).T
        x, y, z = X
        i, j, k = F

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightpink', opacity=0.50))

    # Make 10th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []

    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Weight at index %d: " % (mod_place)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=50,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[np.min(X_avg[0]) - 100, np.max(X_avg[0]) + 100], ),
            yaxis=dict(range=[np.min(X_avg[1]) - 100, np.max(X_avg[1]) + 100], ),
            zaxis=dict(range=[np.min(X_avg[2]) - 100, np.max(X_avg[2]) + 100], ),
            aspectratio=dict(x=((np.max(X_avg[0]) + 100) - (np.min(X_avg[0]) - 100)) / 1000,
                             y=((np.max(X_avg[1]) + 100) - (np.min(X_avg[1]) - 100)) / 1000,
                             z=((np.max(X_avg[2]) + 100) - (np.min(X_avg[2]) - 100)) / 1000)
        ),
        width=1000,
        height=700,

        sliders=sliders

    )

    fig.show()


def create_new_car(features, X_avg, weights_all, n_features=191):
    # Add traces, one for each slider step
    mod_weights = np.zeros(len(features[0]))
    for mod_place in range(n_features):
        fig = go.Figure()
        for step in np.arange(0, 1.05, 0.05):
            normalize = np.max(weights_all[mod_place]) - np.min(weights_all[mod_place])
            mod_weights[mod_place] = step * normalize + np.min(weights_all[mod_place])

            new_shape = np.add(X_avg, np.matmul(features, (mod_weights)).reshape(-1, 3).T)

            fig.add_trace(
                go.Scatter3d(
                    visible=False,
                    name="𝜈 = " + str(step),
                    x=new_shape[0],
                    y=new_shape[1],
                    z=new_shape[2],
                    mode='markers', marker=dict(
                        size=1,
                        color=new_shape[2],  # set color to an array/list of desired values
                        colorscale='Viridis'
                    )
                )

            )

        # Make 10th trace visible
        fig.data[0].visible = True

        # Create and add slider
        steps = []

        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Feature No.%d" % (mod_place)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=50,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[np.min(X_avg[0]) - 100, np.max(X_avg[0]) + 100], ),
                yaxis=dict(range=[np.min(X_avg[1]) - 100, np.max(X_avg[1]) + 100], ),
                zaxis=dict(range=[np.min(X_avg[2]) - 100, np.max(X_avg[2]) + 100], ),
                aspectratio=dict(x=((np.max(X_avg[0]) + 100) - (np.min(X_avg[0]) - 100)) / 1000,
                                 y=((np.max(X_avg[1]) + 100) - (np.min(X_avg[1]) - 100)) / 1000,
                                 z=((np.max(X_avg[2]) + 100) - (np.min(X_avg[2]) - 100)) / 1000)
            ),
            width=1000,
            height=700,

            sliders=sliders

        )

        fig.show()

        ratio = int(input("What's your desired weight for this feature? (pick between 0 ~ 20)")) / 20

        normalize = np.max(weights_all[mod_place]) - np.min(weights_all[mod_place])
        mod_weights[mod_place] = ratio * normalize + np.min(weights_all[mod_place])
        clear_output(wait=True)

        end = int(input("Are you finished? 0: No, 1: Yes"))
        if (end == 1):
            break

    new_shape = np.add(X_avg, np.matmul(features, (mod_weights)).reshape(-1, 3).T)
    draw3DPoints(new_shape, "Congratulation! This is your unique car!")