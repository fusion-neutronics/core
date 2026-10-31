import yamc
import numpy as np

s1 = yamc.Plane(axis="x", offset=2.1)
s2 = yamc.Plane(axis="x", offset=-2.1)
s3 = yamc.Sphere(x0=0, y0=0, z0=0, radius=1)
# s3 = yamc.Cylinder(axis="z", x0=0, y0=0, radius=1)

region1 = s1.below & s2.above & s3.below
inside = region1.contains((0, 0, 0))
print("Point (0, 0, 0) inside region1 = ", inside)
print(region1.bounding_box())

map = region1.sample_slice(resolution=500)
print(np.array(map))

# import matplotlib.pyplot as plt
# plt.imshow(map)

# plt.show()