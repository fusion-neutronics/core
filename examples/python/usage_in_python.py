import numpy as np
import yamc


s1 = yamc.Plane(axis="z", offset=5)
s2 = yamc.Sphere(x0=0,y0=0,z0=1, radius=3)
s3 = yamc.Cylinder(axis="z", x0=0, y0=0, radius=1)

region1 = s1.below & s2.above | ~s3.below

inside = region1.contains((0, 0, 0))

print("Point inside region1?", inside)

s4 = yamc.Plane(axis="x", offset=1.0)
region2 = s2.below

inside = region2.contains((0, 0, 0))

print("Point inside region2?", inside)

bb = region2.bounding_box()
print("Bounding box of region2:", bb.lower_left, bb.upper_right)

print(f'Bounding box center {bb.center}')

print(f"bb width {bb.width}")



results = []
for x in np.linspace(
    bb.lower_left[0], bb.upper_right[0], 10
):
    for y in np.linspace(
        bb.lower_left[1], bb.upper_right[1], 10
    ):
        contains = region2.contains((x, y, 0))
        # print(f"Point ({x}, {y}, 0) inside region2? {contains}")
        results.append(int(contains))

results_np = np.array(results).reshape((10, 10))
print(results_np)
# import matplotlib.pyplot as plt

# plt.imshow(results_np)
# plt.show()