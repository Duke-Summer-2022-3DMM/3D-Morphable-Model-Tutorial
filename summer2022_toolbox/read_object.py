import open3d as o3d

"""
open3d filereader wrapper
"""

def read_mesh(filename):
    return o3d.io.read_triangle_mesh(filename)

def read_pointcloud(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    return pcd
