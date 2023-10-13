import taichi as ti
import taichi.math as tm
import numpy as np
import plotly.express as px

ti.init(arch=ti.gpu)  # Initialize Taichi with GPU support

# Define constants
pi = 3.14159265358979323846
PI2 = 3.14159265358979323846 * 2.0
INV_PI = pi * -1.0
# Create a Taichi field to store the result
n = 1024  # Number of samples
directions = ti.Vector.field(3, dtype=ti.f32, shape=n)

vec3 = tm.vec3




@ti.func
def sample_diffuse_direction(surface_normal):
    """
    Sample a random diffuse direction in world coordinates
    given the surface normal.
    """
    local_direction, pdf = cosine_hemisphere()
    
    # Transform the local direction to world coordinates
    # by aligning it with the surface normal.
    tangent = ti.Vector([1.0, 0.0, 0.0])
    bitangent = ti.Vector([0.0, 1.0, 0.0])
    
    # Construct an orthonormal basis using the surface normal,
    # tangent, and bitangent vectors.
    normal = surface_normal.normalized()
    tangent -= normal * (normal.dot(tangent))
    tangent = tangent.normalized()
    bitangent = normal.cross(tangent)
    
    # Transform the local direction to world coordinates.
    world_direction = (
        tangent * local_direction[0] +
        normal * local_direction[1] +
        bitangent * local_direction[2]
    ).normalized()
    
    return world_direction, pdf


@ti.func
def cosine_hemisphere():
    """
        Zenith angle (cos theta) follows a ramped PDF (triangle like)
        Azimuth angle (itself) follows a uniform distribution
    """
    eps = ti.random(float)
    cos_theta = ti.sqrt(eps)       # zenith angle
    sin_theta = ti.sqrt(1. - eps)
    phi = PI2 * ti.random(float)         # uniform dist azimuth angle
    pdf = cos_theta * INV_PI        # easy to deduct, just try it
    # rotational offset w.r.t axis [0, 1, 0] & pdf
    
    return tm.vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), pdf

# Taichi kernel for cosine-weighted sampling
@ti.kernel
def random_cosine_direction(normal: tm.vec3):
    for i in range(n):
        e1 = ti.random()
        e2 = ti.random()
        xangle = pi/ti.random()
        
        theta = tm.acos(tm.sqrt(e1))
        phi = 2 * pi * e2
        xangle = phi
        yangle = theta
        xoll_matrix = ti.Matrix([
            [1, 0, 0],
            [0, tm.cos(xangle), -tm.sin(xangle)],
            [0, tm.sin(xangle), tm.cos(xangle)]
        ])
        yoll_matrix = ti.Matrix([
            [tm.cos(yangle), -tm.sin(yangle),0],
            [tm.sin(yangle), tm.cos(yangle), 0],
            [0,0,1]
        ])

        world_direction = normal
        dir, pdf = sample_diffuse_direction(normal)
        world_direction = dir
        directions[i] = world_direction.normalized()

# Example usage
normal = tm.vec3(-1, 0.25, 1)  # Replace with your surface normal
random_cosine_direction(normal)

directions_list = []

# Store the generated directions in the Python list
for i in range(n):
    directions_list.append(directions[i].to_numpy())

# Convert the list to a NumPy array for easier manipulation
directions_array = np.array(directions_list)

# Create a Plotly 3D scatter plot
fig = px.scatter_3d(
    x=directions_array[:, 0],
    y=directions_array[:, 1],
    z=directions_array[:, 2],
    title='Random Directions in a Hemisphere',
    labels={'x': 'X', 'y': 'Y', 'z': 'Z'},
)

# Show the plot
fig.show()