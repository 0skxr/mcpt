import taichi.math as tm
import math
import taichi as ti
import random
from voxypy.models import Entity, Voxel
import PIL.Image as im


ti.init(arch=ti.vulkan)

vec3 = ti.math.vec3




entity = Entity().from_file("ffp\\tree.vox")


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

@ti.dataclass
class hit:
    d: ti.types.f32
    cord: tm.vec3
    normal: vec3


@ti.func
def ray_color(ray_origin, ray_direction):
    t = 0.5 * (tm.normalize(ray_direction).y + 1)
    return ((1.0 - t) * tm.vec3(1,1,1) + t * tm.vec3(0.5, 0.7, 1.0))
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

texture = ti.field(dtype=tm.vec3, shape=(16, 16))

im = im.open('ffp\grass_block_side.png')
rgb_im = im.convert('RGB')



for x in range(16):
    for y in range(16):
        r, g, b = rgb_im.getpixel((x, y))
        texture[x,y] = vec3(r,g,b) / 255



voxel_arr = []

s = 10000

voxels = Voxel.field(shape=(s))




print(zmax)

palette = entity.get_palette()

chunk = ti.field(dtype=ti.types.i32, shape=(xmax,zmax,ymax))
color = ti.field(dtype=vec3, shape=(xmax,zmax,ymax))
dmap = ti.field(dtype=vec3, shape=(xmax,zmax,ymax))

for x in range(xmax):
    for y in range(ymax):
        for z in range(zmax):
            if(entity.get(x,y,z) != 0):
                chunk[x,z,y] = 1
                f = vec3(palette[entity.get(x,y,z)._color][0],palette[entity.get(x,y,z)._color][1],palette[entity.get(x,y,z)._color][2])/255
                color[x,z,y] = f
                print(str(x) + " " + str(y) + " " + str(z))


print("loaded")
@ti.func
def get_cube_normal(nx, ny, nz):
    max_comp = tm.max(abs(nx), abs(ny), abs(nz))
    normal = vec3(0)
    if max_comp == abs(nx):
        normal = vec3(tm.sign(nx), 0, 0)
    elif max_comp == abs(ny):
        normal = vec3(0, tm.sign(ny), 0)
    elif max_comp == abs(nz):
        normal = vec3(0, 0, tm.sign(nz))
    return normal


@ti.func
def scene_sdf(p):
    min_dist = inf

    for x in range(tm.clamp(int(p.x)-16,0,xmax),tm.clamp(int(p.x)+16,0,xmax)):
        for y in range(tm.clamp(int(p.y)-16,0,xmax),tm.clamp(int(p.y)+16,0,xmax)):
            for z in range(tm.clamp(int(p.z)-16,0,xmax),tm.clamp(int(p.z)+16,0,xmax)):
                if chunk[x, y, z] > 0:
                    voxel_min = vec3(x, y, z) - 0.5
                    voxel_max = vec3(x, y, z) + 0.5
                    dist_x = tm.max(voxel_min.x - p.x, p.x - voxel_max.x)
                    dist_y = tm.max(voxel_min.y - p.y, p.y - voxel_max.y)
                    dist_z = tm.max(voxel_min.z - p.z, p.z - voxel_max.z)

                    # Use the maximum of the component-wise distances to get the signed distance
                    dist = tm.max(dist_x, tm.max(dist_y, dist_z))

                    if dist < min_dist:
                        min_dist = dist

    return min_dist


@ti.func
def intersection(o, d):
    max_distance = inf 
    distance = 0.1
    min_dist = inf
    p = vec3(0)
    nx = 0
    ny = 0
    nz = 0
    normal = vec3(0)
    p2 = vec3(0)
    for i in range(48000):
        p = o + d * distance
        nx = int(tm.round(p.x))
        ny = int(tm.round(p.y))
        nz = int(tm.round(p.z))
        cur_dist = 0.0
        if 0 <= nx < xmax and 0 <= ny < ymax and 0 <= nz < zmax:
            if chunk[nx, ny, nz] == 1:
                cur_dist = tm.distance(o,p)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    normal = (vec3(nx,ny,nz) - (p + (d * -1 * 0.1)))
                    normal=get_normal(normal) * -1
                    p2 = p
                    break
        else:
            break
                    
        distance += 0.01 # Reduce the step size for ray marching
        if distance >= max_distance:
            break

    # Normale normieren (Länge 1)
    normal = tm.normalize(normal)

    return hit(cord=p2, d=min_dist, normal=normal)




@ti.func
def get_normal(surface_normal):
    maximum = 0.0
    index = 0
    for i in range(3):
        if ti.abs(surface_normal[i]) > ti.abs(maximum):
            index = i
            maximum = surface_normal[i]
    surface_normal = vec3(0)
    surface_normal[index] = maximum
    return surface_normal


pixels = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
blur = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
final = ti.field(dtype=tm.vec3, shape=(image_width, image_height))

albedo = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
normals = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
depth = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
diffuse = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
hit_point = ti.field(dtype=tm.vec3, shape=(image_width, image_height))

@ti.func
def gbuff(o,d,i,j):
    hitrecord = intersection(o=o, d=d)
    if hitrecord.d > 0 and hitrecord.d != inf:
        x = int(ti.round(hitrecord.cord.x))
        y = int(ti.round(hitrecord.cord.y))
        z = int(ti.round(hitrecord.cord.z))
        f = vec3(x-hitrecord.cord.x+0.5,y-hitrecord.cord.y+0.5,z-hitrecord.cord.z+0.5)*16
        albedo[i,j] = texture[int(f.x),int(f.y)]
        normal = hitrecord.normal

        normals[i,j] = hitrecord.normal
        depth[i,j] = hitrecord.d 
        hit_point[i,j] = hitrecord.cord
    else:
        c = ray_color(ray_origin=o, ray_direction=d)  
        depth[i, j] = vec3(0)
        normals[i, j] = vec3(0)
        albedo[i,j] = vec3(inf)
        hit_point[i,j] = vec3(0)

@ti.func
def render_diffuse(o, d, i, j):
    c = vec3(1)
    normal = normals[i, j]
    o = hit_point[i,j]
    d = (random_in_unit_sphere() + normal)
    if(depth[i,j].x > 0):
        for b in range(10):
            hitrecord = intersection(o=o, d=d)
            if hitrecord.d > 0 and hitrecord.d != inf:
                if(b == 9):
                    c = vec3(0)
                else:
                    x = int(ti.round(hitrecord.cord.x))
                    y = int(ti.round(hitrecord.cord.y))
                    z = int(ti.round(hitrecord.cord.z))
                    f = vec3(x - hitrecord.cord.x + 0.5, y - hitrecord.cord.y + 0.5, z - hitrecord.cord.z + 0.5) * 16
                    c = c * (texture[int(f.x), int(f.y)]) * (1 / hitrecord.d*hitrecord.d)
                    normal = hitrecord.normal                    
                    o = hitrecord.cord + (normal * 0.01)  # Move the origin to the hit point
                    d = ((random_in_unit_sphere() + normal))
                    d = tm.normalize(d)  # Generate a random direction for the next ray
            else:
                c = c * ray_color(ray_origin=o,ray_direction=d)
                break
    else:
        c = vec3(0)
    return c

@ti.func
def render_specular(o, d, i, j):
    c = vec3(1)
    normal = normals[i, j]
    o = hit_point[i,j]
    d = tm.reflect(x=d,n=normal)
    if(depth[i,j].x > 0):
        for b in range(24):
            hitrecord = intersection(o=o, d=d)
            if hitrecord.d > 0 and hitrecord.d != inf:
                if(b == 23):
                    c = vec3(0)
                else:
                    x = int(ti.round(hitrecord.cord.x))
                    y = int(ti.round(hitrecord.cord.y))
                    z = int(ti.round(hitrecord.cord.z))
                    f = vec3(x - hitrecord.cord.x + 0.5, y - hitrecord.cord.y + 0.5, z - hitrecord.cord.z + 0.5) * 16
                    c = c * (texture[int(f.x), int(f.y)]) * (1 / hitrecord.d*hitrecord.d)
                    normal = hitrecord.normal                    
                    o = hitrecord.cord + (normal * 0.01)  # Move the origin to the hit point
                    d = tm.reflect(x=d,n=normal)
            else:
                c = c * ray_color(ray_origin=o,ray_direction=d)
                break
    else:
        c = vec3(0)
    return c



@ti.func
def trace(o, d, i, j):
    c = vec3(1)
    normal = vec3(0)
    for b in range(1):
        hitrecord = intersection(o=o, d=d)
        if hitrecord.d > 0 and hitrecord.d != inf:
            x = int(ti.round(hitrecord.cord.x))
            y = int(ti.round(hitrecord.cord.y))
            z = int(ti.round(hitrecord.cord.z))
            #c = c * color[x,y,z]
            f = vec3(x-hitrecord.cord.x+0.5,y-hitrecord.cord.y+0.5,z-hitrecord.cord.z+0.5)*16
            c = c * texture[int(f.x),int(f.y)]
            normal = hitrecord.normal
            o = hitrecord.cord + (normal * 0) # Move the origin to the hit point
            d =  (tm.reflect(x=d,n=normal) * 1) + ((random_in_unit_sphere() + normal) * 0.1)
            d = tm.normalize(d)  # Generate a random direction for the next ray
            if(b==0):
                normals[i,j] = normal
                depth[i,j] = (16 - hitrecord.d) / 16 
        else:
            c = c * ray_color(ray_origin=o, ray_direction=d)  
            if(b==0):
                depth[i, j] = vec3(0)
                normals[i, j] = vec3(0)
            break
    return c




@ti.kernel
def paint(o: vec3,w: int):
    for i, j in pixels:
        u = i / (image_width - 1)
        v = j / (image_height - 1)
        d =  lower_left_corner + u * horizontal + v * vertical 
        dif = vec3(0)
        gbuff(o=o,d=d,i=i,j=j)
        for s in range(2):
            dif += render_diffuse(o=o,d=d,i=i,j=j) 
        diffuse[i,j] = dif / 2 + render_specular(o=o,d=d,i=i,j=j) * 0.4
    for i, j in pixels:
        pixels[i,j] = albedo[i,j] * diffuse[i,j]
    
    
gui = ti.GUI("render", res=(image_width, image_height),fast_gui=True)
diffuse_gui = ti.GUI("diffuse", res=(image_width, image_height),fast_gui=False)

w = 1



print(xmax)
print(ymax)
print(zmax)
while gui.running:
    gui.get_event()  # must be called before is_pressed
    if gui.is_pressed('a', ti.GUI.LEFT):
        origin.x += - 0.1
    elif gui.is_pressed('d', ti.GUI.RIGHT):
        origin.x += 0.1
    elif gui.is_pressed('w', ti.GUI.UP):
        origin.z += - 0.1
    elif gui.is_pressed('s', ti.GUI.DOWN):
        origin.z += 0.1
    
    origin.y = 2
    gui.show()
    paint(origin,w=w+1)
    gui.set_image(pixels)
    gui.show()
    diffuse_gui.set_image(diffuse)
    diffuse_gui.show()
    w += 1