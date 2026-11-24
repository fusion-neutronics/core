import yamc
import matplotlib.pyplot as plt


s1 = yamc.Plane(axis="x", offset=2.1)
s2 = yamc.Plane(axis="x", offset=-2.1)
s3 = yamc.Sphere(x0=0, y0=0, z0=0, radius=4.2)
# s3 = yamc.Cylinder(x0=0, y0=0, z0=0, axis_x=0, axis_y=0, axis_z=1, radius=1)

surfaces_dict = {s1.id: s1, s2.id: s2, s3.id: s3}

region1 = s1.below & s2.above & s3.below
inside = region1.contains((0, 0, 0))
print("Point inside region1?", inside)
print(region1.bounding_box())


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


results_cell_id = region2.sample_slice(
    origin=bb.center,
    # width=(bb.width[0], bb.width[1]), found automatically
    resolution=40000,
    basis='xy'
)

plt.imshow(results_cell_id)
plt.show()