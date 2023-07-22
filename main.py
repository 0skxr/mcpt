import taichi.math as tm
import math
import taichi as ti
import random
from voxypy.models import Entity, Voxel


ti.init(arch=ti.gpu)

vec3 = ti.math.vec3


entity = Entity().from_file("ffp\\test.vox")


print(entity.get(1,1,1))

xmax = 0

for x in range(len(entity.all_voxels())):
    try:
        (entity.get(x,0,0))
    except:
        xmax = x
        break
ymax = 0

for y in range(len(entity.all_voxels())):
    try:
        (entity.get(0,y,0))
    except:
        ymax = y
        break
zmax = 0

for z in range(len(entity.all_voxels())):
    try:
        (entity.get(0,0,z))
    except:
        zmax = z
        break



@ti.func
def get_hitpoint(o, d, t):
    hitpoint = o + t * d
    return hitpoint

@ti.dataclass
class Voxel:
    min: vec3
    max: vec3
    color: vec3
    ref: bool

@ti.func
def ray_color(ray_origin, ray_direction):
    t = 0.5 * (tm.normalize(ray_direction).y + 1)
    return ((1.0 - t) * tm.vec3(1,1,1) + t * tm.vec3(0.5, 0.7, 1.0)) * 1
    #return vec3(0)


@ti.func
def random_in_unit_sphere():
  theta = 2.0 * 3.14 * ti.random()
  phi = ti.acos((2.0 * ti.random()) - 1.0)
  r = ti.pow(ti.random(), 1.0 / 3.0)
  return ti.Vector([r * ti.sin(phi) * ti.cos(theta), r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi)])



eps = 1e-4
inf = 1e10

image_width = 512
image_height = 512

aspect = image_width/image_height

viewport_height = 2
viewport_width = aspect * viewport_height
focal_length = 2

origin = tm.vec3(0.0, 0.0, 0.0)
horizontal = tm.vec3(viewport_width, 0, 0)
vertical = tm.vec3(0, viewport_height, 0)
lower_left_corner = origin - horizontal / 2 - vertical / 2 - tm.vec3(0, 0, focal_length)


pixels = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
blur = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
final = ti.field(dtype=tm.vec3, shape=(image_width, image_height))

normals = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
depth = ti.field(dtype=tm.vec3, shape=(image_width, image_height))

voxel_arr = []

s = 10000

voxels = Voxel.field(shape=(s))

ci = 0
palette = entity.get_palette()
for x in range(xmax):
    for y in range(ymax):
        for z in range(zmax):
            if(entity.get(x,y,z)!=0):
                
                
                farbe = entity.get(x,y,z)._color
                #print(entity.get_palette())
                f = vec3(palette[entity.get(x,y,z)._color][0],palette[entity.get(x,y,z)._color][1],palette[entity.get(x,y,z)._color][2])/270
                voxels[ci]=Voxel(min=vec3(-0.5, -0.5, -0.5)+vec3(x,z,y),max=vec3(0.5, 0.5, 0.5)+vec3(x,z,y),color=(f))
                #print(f)
                ci += 1
                s = ci




@ti.func
def get_normal(voxel, hitpoint):
    surface_normal = hitpoint - (voxel.max - 0.5)
    maximum = 0.0
    index = 0
    for i in range(3):
        if ti.abs(surface_normal[i]) > ti.abs(maximum):
            index = i
            maximum = surface_normal[i]
    surface_normal = vec3(0)
    surface_normal[index] = maximum
    return tm.normalize(surface_normal)


@ti.func
def ray_aabb_intersection(o, d, n):
    near_int = -inf
    far_int = inf

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < voxels[n].min[i] or o[i] > voxels[n].max[i]:
                pass
        else:
            i1 = (voxels[n].min[i] - o[i]) / d[i]
            i2 = (voxels[n].max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    hit = near_int < far_int
    near_int = inf if (near_int <= 0) else near_int
    distance = near_int if hit else inf

    return distance

@ti.func
def trace(o, d,i,j):
    c = vec3(1)
    normal = vec3(0)
    for b in range(20):
        oh = inf
        t = 10000

        for n in range(s):
            h = ray_aabb_intersection(o=o, d=d, n=n)
            if h < oh:
                oh = h
                t = n
        if t != 10000:
            if (voxels[t].ref):
                c = c * voxels[t].color
                o = get_hitpoint(o=o, d=d, t=oh) + 0.0001
                normal = get_normal(voxel=voxels[t], hitpoint=o)
                d = tm.reflect(x=d, n=normal)
            else:
                c = c * voxels[t].color
                o = get_hitpoint(o=o, d=d, t=oh) + 0.0001
                normal = get_normal(voxel=voxels[t], hitpoint=o)
                d = normal + random_in_unit_sphere()
            if(b  == 0 ):
                normals[i,j] = normal
                depth[i,j] = vec3(oh / 100)
        else:
            c = c * ray_color(ray_origin=o,ray_direction=d)
            break
    return c

@ti.kernel
def paint(o: vec3,w: int):
    for i, j in buff:
        buff[i,j] = blur[i,j]
    for i, j in pixels:
        u = i / (image_width - 1)
        v = j / (image_height - 1)
        d =  lower_left_corner + u * horizontal + v * vertical 
        pixels[i,j] = trace(o=o,d=d,i=i,j=j)
    for i, j in pixels:
        blur[i,j] = (pixels[i-1,j]*2 + pixels[i+1,j]*2 + pixels[i,j-1]*2 + pixels[i,j-1]*2 + pixels[i,j]*2)/5
    for i, j in pixels:
        blur[i,j] = (blur[i,j] + buff[i,j]) / 2
        final[i,j] = final[i,j] + blur[i,j]
    for i, j in pixels:
        pixels[i,j] = final[i,j] / w
    


gui = ti.GUI("render", res=(image_width, image_height),fast_gui=True)
n_gui = ti.GUI("normals", res=(image_width, image_height),fast_gui=True)
d_gui = ti.GUI("depth", res=(image_width, image_height),fast_gui=True)
for w in range(100000):
    origin.x = 1.5
    origin.y = 1.5
    origin.z = 8
    paint(origin,w=w+1)
    print(w)
    gui.set_image(pixels)
    gui.show()
    n_gui.set_image(normals)
    n_gui.show()
    d_gui.set_image(depth)
    d_gui.show()