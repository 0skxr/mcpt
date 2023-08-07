import taichi.math as tm
import math
import taichi as ti
import random
from voxypy.models import Entity, Voxel
import PIL.Image as im

ti.init(arch=ti.opengl)

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
    xyz: vec3


@ti.func
def ray_color(ray_origin, ray_direction):
    t = 0.5 * (tm.normalize(ray_direction).y + 1)
    return (((1.0 - t) * tm.vec3(1,1,1) + t * tm.vec3(0.5, 0.7, 1.0))) 
    #return vec3(0)

@ti.func
def random_in_unit_sphere():
    theta = 2.0 * 3.14 * ti.random()
    phi = ti.acos((2.0 * ti.random()) - 1.0)

    # Cosine-weighting factor
    cos_weight = ti.sqrt(ti.random())  

    r = ti.pow(ti.random(), 1.0 / 3.0)
    return ti.Vector([r * cos_weight * ti.sin(phi) * ti.cos(theta), 
                      r * cos_weight * ti.sin(phi) * ti.sin(theta), 
                      r * ti.cos(phi)])




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

texture = ti.field(dtype=tm.vec3, shape=(48, 16))

img = im.open('ffp\\blocks\\emerald_block.png')
rgb_im = img.convert('RGB')

for x in range(16):
    for y in range(16):
        r, g, b = rgb_im.getpixel((x, y))
        texture[x,y] = vec3(r,g,b) / 255
        

img = im.open('ffp\\blocks\\emerald_block_mer.png')
rgb_im = img.convert('RGB')

for x in range(16):
    for y in range(16):
        r, g, b = rgb_im.getpixel((x, y))
        texture[x+16,y] = vec3(r,g,b) / 255
        
        
img = im.open('ffp\sand_n.png')
rgb_im = img.convert('RGB')

for x in range(16):
    for y in range(16):
        r, g, b = rgb_im.getpixel((x, y))
        texture[x+32,y] = vec3(r,g,b) / 255



voxel_arr = []

s = 10000

voxels = Voxel.field(shape=(s))




print(zmax)

#palette = entity.get_palette()

chunk = ti.field(dtype=ti.types.i32, shape=(xmax,zmax,ymax))
color = ti.field(dtype=vec3, shape=(xmax,zmax,ymax))
dmap = ti.field(dtype=vec3, shape=(xmax,zmax,ymax))

for x in range(xmax):
    for y in range(ymax):
        for z in range(zmax):
            if(entity.get(x,y,z) != 0):
                chunk[x,z,y] = random.randint(1,10)
                #f = vec3(palette[entity.get(x,y,z)._color][0],palette[entity.get(x,y,z)._color][1],palette[entity.get(x,y,z)._color][2])/255
                #color[x,z,y] = f
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
def intersection(o, d):
    d = tm.normalize(d)
    X = int(tm.round(o.x))
    Y = int(tm.round(o.y))
    Z = int(tm.round(o.z))

    stepX = int(tm.sign(d.x))
    stepY = int(tm.sign(d.y))
    stepZ = int(tm.sign(d.z))

    tMaxX = (X + (0.5 if stepX > 0 else -0.5) - o.x) / d.x
    tMaxY = (Y + (0.5 if stepY > 0 else -0.5) - o.y) / d.y
    tMaxZ = (Z + (0.5 if stepZ > 0 else -0.5) - o.z) / d.z
    tMax = 0.0
    tDeltaX = abs(1 / d.x)
    tDeltaY = abs(1 / d.y)
    tDeltaZ = abs(1 / d.z)

    voxel = 0
    D = 0.0 

    for i in range(100):
        if tMaxX < tMaxY and tMaxX < tMaxZ:
            tMax = tMaxX
            X += stepX
            tMaxX += tDeltaX
            if X == xmax or X < 0:
                break
        elif tMaxY < tMaxZ:
            tMax = tMaxY
            Y += stepY
            tMaxY += tDeltaY
            if Y == ymax or Y < 0:
                break
        else:
            tMax = tMaxZ
            Z += stepZ
            tMaxZ += tDeltaZ
            if Z == zmax or Z < 0:
                break

        voxel = chunk[X, Y, Z] if 0 <= X < xmax and 0 <= Y < ymax and 0 <= Z < zmax else 0
        if voxel > 0:
            break

    if voxel > 0:
        D = tMax
    else:
        D = 0

    return hit(cord=o+d*(D-0.001), d=D, normal=get_normal((o+d*D)-vec3(X,Y,Z)),xyz=vec3(X,Y,Z))

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
    return tm.normalize(surface_normal)


pixels = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
blur = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
final = ti.field(dtype=tm.vec3, shape=(image_width, image_height))

albedo = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
specular = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
specular_buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
normals = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
depth = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
diffuse = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
hit_point = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
sky = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
rd = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
@ti.func
def texture_map(cord, normal,offset,spec):
    x = int(ti.round(cord.x))
    y = int(ti.round(cord.y))
    z = int(ti.round(cord.z))
    f = vec3(x - cord.x, y - cord.y, z - cord.z)
    sign = tm.sign(f)
    f = ti.abs((f + 0.5) * 16)
    s = 1.0
    if(ti.abs(normal.z) > 0):
        f = (texture[int(f.x+offset), int(f.y)])
        if(spec):
            s = tm.sign(normal.z)
    elif(ti.abs(normal.y) > 0):
        f = (texture[int(f.x+offset), int(f.z)])
        if(spec):
            s = tm.sign(normal.y)
    else:
        f = (texture[int(f.z+offset), int(f.y)])
        if(spec):
            s = tm.sign(normal.x)

    return f * s





@ti.func
def gbuff(o,d,i,j):
    hitrecord = intersection(o=o, d=d)
    if hitrecord.d > 0 and hitrecord.d != inf:
        x = int(hitrecord.xyz.x)
        y = int(hitrecord.xyz.y)
        z = int(hitrecord.xyz.z)
        normal = hitrecord.normal
        albedo[i,j] = texture_map(hitrecord.cord,normal,0,False)
        specular[i,j] = texture_map(hitrecord.cord,normal,16,False)

        #albedo[i,j] = (chunk[x,y,z]==3)/5
        normals[i,j] = normal
        depth[i,j] = hitrecord.d 
        hit_point[i,j] = hitrecord.cord
    else:
        c = 0  
        depth[i, j] = vec3(0)
        normals[i, j] = vec3(0)
        albedo[i,j] = vec3(inf)
        hit_point[i,j] = vec3(0)
        specular[i,j] = vec3(0)


@ti.func
def bilateral_filter(sigma_s: ti.f32, sigma_r: ti.f32):

    blur_radius_s = ti.ceil(sigma_s * 3, int)

    for i in range(image_width):
        for j in range(image_height):
            k_begin, k_end = max(0, i - blur_radius_s), min(image_width, i + blur_radius_s + 1)
            l_begin, l_end = max(0, j - blur_radius_s), min(image_height, j + blur_radius_s + 1)
            total_rgb = tm.vec3(0.0)
            total_weight = 0.0
            for k in range(k_begin,k_end):
                for l in range(l_begin,l_end):
                    dist = ((i - k)**2 + (j - l)**2) / sigma_s**2 + (diffuse[i, j].cast(ti.f32) - diffuse[k, l].cast(ti.f32)).norm_sqr() / sigma_r**2
                    w = ti.exp(-0.5 * dist)
                    total_rgb += diffuse[k, l] * w
                    total_weight += w

            blur[i, j] = (total_rgb / total_weight)

@ti.func
def lambertian_brdf(incoming_direction, outgoing_direction, surface_normal, diffuse_reflectance):
    incoming_direction = tm.normalize(incoming_direction)
    outgoing_direction = tm.normalize(outgoing_direction)
    surface_normal = tm.normalize(surface_normal)

    # Compute the cosine of the incident angle (angle between light direction and surface normal)
    cos_incident_angle = tm.max(0, tm.dot(incoming_direction, surface_normal))

    # Compute the Lambertian BRDF value
    brdf_value = (diffuse_reflectance / 3.14) * cos_incident_angle

    return brdf_value






    
@ti.func
def render_diffuse(o, d, i, j):
    c = vec3(1)
    normal = normals[i, j]
    spec = specular[i,j].x
    o = hit_point[i,j] + (normal * 0.02)
    r = random_in_unit_sphere()
    if tm.dot(normal, r) < 0:
        r = r - 2 * tm.dot(normal, r) * normal
    roughness = tm.pow(1 - spec, 2)
    d = r #* (roughness) + (tm.reflect(x=d,n=normal) * (1-roughness))
    d = tm.normalize(d)
    if(depth[i,j].x > 0):
        for b in range(10):
            if(b == 9):
                c = vec3(0) 
            else:
                hitrecord = intersection(o=o, d=d)
                if hitrecord.d > 0 and hitrecord.d != inf:
                    x = int(hitrecord.xyz.x)
                    y = int(hitrecord.xyz.y)
                    z = int(hitrecord.xyz.z)
                    f = vec3(x - hitrecord.cord.x + 0.5, y - hitrecord.cord.y + 0.5, z - hitrecord.cord.z + 0.5) * 16
                    nc = texture_map(hitrecord.cord,normal,0,False)
                    mer = texture_map(hitrecord.cord,normal,16,False)
                    normal = hitrecord.normal                    
                    o = hitrecord.cord + (normal * 0.02)  # Move the origin to the hit point
                    r = random_in_unit_sphere()
                    d = r  
                    if tm.dot(normal, r) < 0:
                        r = r - 2 * tm.dot(normal, r) * normal
                    d = tm.normalize(d)
                    c = c * (nc * (1+mer.y*16))
                else:
                    c = c * (ray_color(ray_origin=o,ray_direction=d)) # * lambertian_brdf(incoming_direction=id,outgoing_direction=d,surface_normal=normal,diffuse_reflectance=roughness))
                    break
    else:
        c = vec3(0)
    return c

@ti.func
def render_specular(o, d, i, j):
    c = vec3(1)
    normal = normals[i, j]
    o = hit_point[i,j] + (normal * 0.02)
    r = random_in_unit_sphere()
    if tm.dot(normal, r) < 0:
        r = r - 2 * tm.dot(normal, r) * normal
    d = tm.reflect(n=tm.normalize(normal),x=d)
    d = d*(1-specular[i,j].z) + r*specular[i,j].z
    d = tm.normalize(d)
    if(depth[i,j].x > 0):
        for b in range(10):
            if(b == 9):
                c = vec3(0)
            else:
                hitrecord = intersection(o=o, d=d)
                if hitrecord.d > 0 and hitrecord.d != inf:
                    x = int(hitrecord.xyz.x)
                    y = int(hitrecord.xyz.y)
                    z = int(hitrecord.xyz.z)
                    f = vec3(x - hitrecord.cord.x + 0.5, y - hitrecord.cord.y + 0.5, z - hitrecord.cord.z + 0.5) * 16
                    nc = texture_map(hitrecord.cord,normal,0,False)
                    id = d
                    mer = texture_map(hitrecord.cord,normal,16,False)
                    normal = hitrecord.normal                    
                    o = hitrecord.cord + (normal * 0.02)  # Move the origin to the hit point
                    r = random_in_unit_sphere()
                    d = r  #* (1-roughness)  + (tm.reflect(x=d,n=normal) * (1-roughness))
                    if tm.dot(normal, r) < 0:
                        r = r - 2 * tm.dot(normal, r) * normal
                    d = tm.normalize(d)
                    c = c * (nc * (1+mer.y*16)) # lambertian_brdf(incoming_direction=id,outgoing_direction=d,surface_normal=normal,diffuse_reflectance=roughness))
                else:
                    c =  c * (ray_color(ray_origin=o,ray_direction=d)) # * lambertian_brdf(incoming_direction=id,outgoing_direction=d,surface_normal=normal,diffuse_reflectance=roughness))
                    break
    else:
        c = vec3(0)
    return c



vel = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
last = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
test = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#

@ti.func
def ACESFilm(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return tm.clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0, 1.0)

@ti.kernel
def paint(o: vec3,w: int, rot: vec3):
    angle = rot.y * (3.14 / 180)  # Rotate 45 degrees

# Create the rotation matrix using the math module
    rot_matrix = ti.Matrix([[tm.cos(angle), 0, tm.sin(angle)],
                            [0.0, 1, 0.0],
                            [-tm.sin(angle), 0.0, tm.cos(angle)]])

    for i, j in pixels:
        u = i / (image_width - 1)
        v = j / (image_height - 1)
        d =  lower_left_corner + u * horizontal + v * vertical 
        d = rot_matrix @ d
        rd[i,j] = d
        gbuff(o=o,d=d,i=i,j=j)
        dif = vec3(0)
        specdif =vec3(0)
        for sa in range(2):
            dif += render_diffuse(o,d,i,j) 
            specdif += render_specular(o,d,i,j)
        diffuse[i,j] =  (dif /2) 
        specular_buff[i,j] = (specdif/2)
        c = vec3(0)
        if (normals[i,j].x == 0 and normals[i,j].y == 0 and normals[i,j].z == 0):
            c = ray_color(ray_direction=d,ray_origin=o)
            specular_buff[i,j] = vec3(0)
        
        sky[i,j] = c
        
        
    for i, j in pixels:
        if(albedo[i,j].x  > 1.99888 and not (normals[i,j].x == 0 and normals[i,j].y == 0 and normals[i,j].z == 0)):
            pixels[i,j] = albedo[i,j]
        else:
            inc = tm.clamp(tm.cos(ti.abs(tm.dot(rd[i,j],normals[i,j]))),0.0,1.0)
            pixels[i,j] = (albedo[i,j] * (diffuse[i,j]* (1-inc) + specular_buff[i,j]*inc)) + last[i,j] / 2 + sky[i,j] + (albedo[i,j]*specular[i,j].y*2)
            #pixels[i,j] =vec3(inc)
    
    for i, j in last:
        last[i,j] = pixels[i,j]
    
    for i, j in pixels:
        pixels[i,j] = ACESFilm(pixels[i,j]*0.5)
gui = ti.GUI("render", res=(image_width, image_height),fast_gui=True)
diffuse_gui = ti.GUI("diffuse", res=(image_width, image_height),fast_gui=True)

w = 1



print(xmax)
print(ymax)
print(zmax)
rot = vec3(0)
while gui.running:
    gui.get_event()  # must be called before is_pressed
    if gui.is_pressed('a', ti.GUI.LEFT):
        rot.y += - 1
    elif gui.is_pressed('d', ti.GUI.RIGHT):
        rot.y += 1
    elif gui.is_pressed('w', ti.GUI.UP):
        origin.z += - 0.1
    elif gui.is_pressed('s', ti.GUI.DOWN):
        origin.z += 0.1
    
    origin.y = 2
    gui.show()
    paint(origin,w=w+1,rot=rot)
    gui.set_image(pixels)
    gui.show()
    diffuse_gui.set_image(specular_buff)
    diffuse_gui.show()
    w += 1