import taichi as ti
import taichi.math as tm

pixels = None
buff = None
blur = None
final = None
albedo = None
specular = None
specular_buff = None
normals = None
depth = None
diffuse = None
hit_point = None
sky = None
rd = None

def init(image_width, image_height):
    """
    Initialize the fields and buffers for the renderer.

    Parameters:
    - image_width (int): The width of the image.
    - image_height (int): The height of the image.
    """
    global pixels, buff, blur, final, albedo, specular, specular_buff, normals, depth, diffuse, hit_point, sky, rd

    pixels = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    blur = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    final = ti.field(dtype=tm.vec3, shape=(image_width, image_height))

    albedo = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    specular = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    specular_buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    normals = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    depth = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    diffuse = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    hit_point = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    sky = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
    rd = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
