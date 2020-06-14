# Point cloud sampling logic: 1.calculate the size of all the triangles in the mesh
#                             2.calculate the density
#                             3.calculate the size of one triangle and the points should be sampled from it
#                             4.sample the points using Generating random points in triangles
import numpy as np
import os
import math


def triangle_size(x, y, z):
    vec1 = x - y
    vec2 = z - y
    return np.linalg.norm(np.cross(vec1, vec2)) / 2


def sampling_point(x, y, z):
    vec1 = x - y
    vec2 = z - y
    a = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    if a + b > 1:
        a = 1 - a
        b = 1 - b
    p = y + vec1 * a + vec2 * b
    return p


total_sample = 100000
source_path = "D:\Download\ModelNet40"
category = "xbox"
subcat = "train"
path = os.path.join(source_path, category, subcat)
os.chdir(path)
print("transforming files in path: " + path)
for filename in os.listdir(path):
    print("current file:" + filename)
    f = open(filename, 'r')
    content = f.readlines()
    count = len(content)
    num_vertice, num_triangle, v3 = content[1].split()
    vertices = content[2:2 + int(num_vertice)]
    triangles = content[2 + int(num_vertice):2 + int(num_vertice) + 1 + int(num_triangle)]
    size_of_triangles = []
    total_size = 0
    for x in range(0, len(triangles)):
        line = triangles[x]
        trinum = line.split()
        trinum = trinum[1:4]
        str1 = vertices[int(trinum[0])]
        str2 = vertices[int(trinum[1])]
        str3 = vertices[int(trinum[2])]
        str1 = str1.split()
        str2 = str2.split()
        str3 = str3.split()
        point1 = []
        point2 = []
        point3 = []
        for i in range(0, 3):
            point1.append(float(str1[i]))
            point2.append(float(str2[i]))
            point3.append(float(str3[i]))
        size = triangle_size(np.array(point1), np.array(point2), np.array(point3))
        size_of_triangles.append(size)
        total_size += size

    print('size of tri:' + str(size_of_triangles))
    print('total size:' + str(total_size))
    density = total_sample / total_size
    size_of_triangles = [x * density for x in size_of_triangles]
    points_per_triangle = [round(x) for x in
                           size_of_triangles]  # list of how many points should be sampled from each triangle
    sample_point_list = []
    for x in range(0, len(triangles)):
        line = triangles[x]
        trinum = line.split()
        trinum = trinum[1:4]
        str1 = vertices[int(trinum[0])]
        str2 = vertices[int(trinum[1])]
        str3 = vertices[int(trinum[2])]
        str1 = str1.split()
        str2 = str2.split()
        str3 = str3.split()
        point1 = []
        point2 = []
        point3 = []
        for i in range(0, 3):
            point1.append(float(str1[i]))
            point2.append(float(str2[i]))
            point3.append(float(str3[i]))
        for i in range(0, int(points_per_triangle[x])):
            p = sampling_point(np.array(point1), np.array(point2), np.array(point3))
            sample_point_list.append(p)
    outname = filename.replace('.off', '') + '.ply'
    fo = open(outname, "w+")
    fo.write("ply\nformat ascii 1.0\n")
    strforline='element vertex '+str(len(sample_point_list))+'\n'
    fo.write(strforline)
    fo.write("property float x\nproperty float y\nproperty float z\nend_header\n")
    for i in range(0,len(sample_point_list)):
        fo.write(str(sample_point_list[i])[1:-1])
        fo.write('\n')
    fo.close()