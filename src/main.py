import taichi.math as tm
import math
import taichi as ti
import random
from voxypy.models import Entity, Voxel
import PIL.Image as im

ti.init(arch=ti.vulkan)
vec3 = ti.math.vec3


entity = Entity().from_file("tree.vox")

print(entity.get(1,1,1))

xmax = 32
ymax = 64
zmax = 32




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
	sky_color = (1.0 - t) * tm.vec3(1, 1, 1) + t * tm.vec3(0.5, 0.7, 1.0)
	sun = vec3(0) 
	sun_pos = vec3(0,0.3,-1) * 100
	sun = sky_color * 0.5
	for i in range(63):
		v_pos = ray_direction * (i*2)
		d = tm.sqrt((sun_pos.x-v_pos.x)**2+(sun_pos.y-v_pos.y)**2+(sun_pos.z-v_pos.z)**2)
		if(d < 25):
			sun = (vec3(201,141,38)/255)*50
			break
	return  sun 

	




eps = 1e-4
inf = 1e10

image_width = 1280
image_height = 720

aspect = image_width/image_height

viewport_height = 2
viewport_width = aspect * viewport_height
focal_length = 2

origin = tm.vec3(0.0, 0.0, 0.0)
horizontal = tm.vec3(viewport_width, 0, 0)
vertical = tm.vec3(0, viewport_height, 0)
lower_left_corner = origin - horizontal / 2 - vertical / 2 - tm.vec3(0, 0, focal_length)




voxel_arr = []

s = 10000

voxels = Voxel.field(shape=(s))




print(zmax)

#palette = entity.get_palette()

chunk = ti.field(dtype=ti.types.i32, shape=(xmax,zmax,ymax))
color = ti.field(dtype=vec3, shape=(xmax,zmax,ymax))
dmap = ti.field(dtype=vec3, shape=(xmax,zmax,ymax))

#for x in range(xmax):
#    for y in range(ymax):
#        for z in range(zmax):
#            if(entity.get(x,y,z) != 0):
#                chunk[x,z,y] = random.randint(1,10)
#                #f = vec3(palette[entity.get(x,y,z)._color][0],palette[entity.get(x,y,z)._color][1],palette[entity.get(x,y,z)._color][2])/255
#                #color[x,z,y] = f
#                print(str(x) + " " + str(y) + " " + str(z))


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
def intersection(o, d, offset):
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
	ofc = 0.0
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
			if(ofc >= offset):
				break
			ofc += 1

	if voxel > 0:
		D = tMax
	else:
		D = 0

	if(X > xmax or Y > ymax or Z > zmax):
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
specular = ti.field(dtype=tm.vec4, shape=(image_width, image_height))
specular_buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
normals = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
depth = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
diffuse = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
hit_point = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
sky = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
rd = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
ref = ti.field(dtype=tm.vec3, shape=(image_width, image_height))
@ti.func


def texture_map(cord, normal,offset,spec,block,depth):
	x = int(ti.round(cord.x ))
	y = int(ti.round(cord.y ))
	z = int(ti.round(cord.z ))
	offset = chunk[int(block.x),int(block.y),int(block.z)] * 16
	f = tm.vec4(x - cord.x, y - cord.y, z - cord.z,0)
	sign = tm.sign(f)
	f = ti.abs((f +0.5) * 16)
	s = 1.0
	if(ti.abs(normal.z) > 0):
		f = (texture[int(f.x+offset), int(f.y),depth])
		if(spec):
			s = tm.sign(normal.z)
	elif(ti.abs(normal.y) > 0):
		f = (texture[int(f.x+offset), int(f.z),depth])
		if(spec):
			s = tm.sign(normal.y)
	else:
		f = (texture[int(f.z+offset), int(f.y),depth])
		if(spec):
			s = tm.sign(normal.x)

	return f



pos = ti.field(dtype=tm.vec3, shape=(image_width, image_height))

@ti.func
def gbuff(o,d,i,j):
	hitrecord = intersection(o=o, d=d,offset=0)
	if hitrecord.d > 0 and hitrecord.d != inf:
		x = int(hitrecord.xyz.x)
		y = int(hitrecord.xyz.y)
		z = int(hitrecord.xyz.z)
		normal = hitrecord.normal
		e = albedo[i,j] = texture_map(hitrecord.cord,normal,0,False,hitrecord.xyz,2).xyz
		albedo[i,j] = texture_map(hitrecord.cord,normal,0,False,hitrecord.xyz,0).xyz * e
		specular[i,j] = texture_map(hitrecord.cord,normal,16,False,hitrecord.xyz,1)
		pos[i,j] = hitrecord.xyz

		#albedo[i,j] = chunk[x,y,z] / 10
		normals[i,j] = normal
		depth[i,j] = hitrecord.d
		hit_point[i,j] = hitrecord.cord
		ref[i,j] = vec3(specular[i,j].w)
	else:
		c = 0
		depth[i, j] = vec3(0)
		normals[i, j] = vec3(0)
		albedo[i,j] = vec3(inf)
		hit_point[i,j] = vec3(0)
		specular[i,j] = tm.vec4(0)
		ref[i,j] = vec3(1)
		pos[i,j] = vec3(inf)

@ti.func
def random_in_unit_sphere():
	E1 = (ti.random()-0.5)*2
	E2 = (ti.random()-0.5)*2
	E3 = (ti.random()-0.5)*2
	return tm.normalize(vec3(E1,E2,E3))



pi = 3.14159265358979323846
PI2 = 3.14159265358979323846 * 2.0
INV_PI = pi * -1.0






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



@ti.func
def sample_diffuse_direction(surface_normal):
	e1 = ti.random()
	e2 = ti.random()
	
	uu = tm.normalize( tm.cross( surface_normal, vec3(0.0,1.0,1.0) ) )
	vv = tm.normalize( tm.cross( uu, surface_normal ) )
	
	ra = tm.sqrt(e2)
	rx = ra*tm.cos(6.2831*e1)
	ry = ra*tm.sin(6.2831*e1)
	rz = tm.sqrt( 1.0-e2 )
	rr = vec3( rx*uu + ry*vv + rz*surface_normal)
	
	return tm.normalize( rr ), 1 #tm.cos(tm.acos(tm.sqrt(e1)))


@ti.func
def render_diffuse(o, d, i, j):
	pdf = 0.0
	d = tm.normalize(d)
	normal = vec3(0)
	c = vec3(1)
	sky = 0
	roughness = tm.pow(1.0 - specular[i,j].x, 2.0)
	roughness = roughness
	spec = specular[i,j].y
	
	c = vec3(1)
	normal = tm.normalize(normals[i, j])
	o = hit_point[i,j] + (normal * 0.02)
 
	randomValue = ti.random()

	doSpecular = (randomValue < spec)

	diffuseRayDir, pdf = sample_diffuse_direction(normal)
	specularRayDir = tm.reflect(x=d, n=normal)

	roughnessSquared = roughness * roughness

	newRayDir = vec3(0)

	if(doSpecular):
		newRayDir = tm.mix(specularRayDir, diffuseRayDir, roughnessSquared)
	else:
		newRayDir = diffuseRayDir

	newRayDir = tm.normalize(newRayDir)
	d = newRayDir
	test[i,j] = roughness * spec
	if(not sky):
		if(depth[i,j].x > 0):
			for b in range(20):
				if(b == 19):
					c = vec3(0)
					#c = c
					break
				else:
					hitrecord = intersection(o=o, d=d,offset=0)
					if hitrecord.d > 0 and hitrecord.d != inf:
						# spec = texture_map(hitrecord.cord,normal,16,False,hitrecord.xyz,1)
						new_c =  texture_map(hitrecord.cord,normal,0,False,hitrecord.xyz,0) * pdf
						e = texture_map(hitrecord.cord,normal,0,False,hitrecord.xyz,2)
						normal = hitrecord.normal
						o = hitrecord.cord + (normal * 0.02)  # Move the origin to the hit point
						r = random_in_unit_sphere()
						d = r  
						if tm.dot(normal, r) < 0:
							r = r - 2 * tm.dot(normal, r) * normal
						d = tm.normalize(d)
						c = c * new_c.xyz * e.x
						if((c.x+c.y+c.z)/3 < 0.05):
							break
						if(e.x == 1):
							pass
						else:
							break
					else:
						c = c * (ray_color(ray_origin=o,ray_direction=d))
						break
		else:
				c = vec3(0)
	return c

@ti.func
def render_ref(o, d, i, j):
	d = tm.normalize(d)
	roughness = tm.pow(1.0 - specular[i,j].x, 2.0)
	roughness = roughness
	roughness = 0
	spec = specular[i,j].y
	spec = 0
	c = vec3(1)
	normal = tm.normalize(normals[i, j])
	o = hit_point[i,j] + (normal * 0.02)
	r = tm.normalize(random_in_unit_sphere())
	return c



vel = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
last = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
test = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#
texture = ti.field(dtype=tm.vec4, shape=(512, 16,5))

acc_buff = ti.field(dtype=tm.vec3, shape=(image_width, image_height))#

@ti.func
def ACESFilm(x):
	a = 2.51
	b = 0.03
	c = 2.43
	d = 0.59
	e = 0.14
	return tm.clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0, 1.0)

@ti.func
def LinearToSRGB(rgb):
	rgb = tm.clamp(rgb, 0.0, 1.0)
	
	return tm.mix(
		tm.pow(rgb, 1.0 / 2.4) * 1.055 - 0.055,
		rgb * 12.92,
		rgb < 0.0031308
	)


@ti.func
def hash1(seed):
	seed += 0.1
	return tm.fract(tm.sin(seed) * 43758.5453123)

@ti.kernel
def paint(o: vec3,w: int, rot: vec3, last_origin: vec3,acc: int):
	
 
	angle_yaw = rot.x
	angle_pitch = rot.y

	offset = vec3(ti.random(),ti.random(),ti.random())
	offset = offset -0.5
	offset = offset * 0.002
# Create the rotation matrix using the math module
	yaw_matrix = ti.Matrix([[tm.cos(angle_yaw), -tm.sin(angle_yaw),0],
							[tm.sin(angle_yaw), tm.cos(angle_yaw), 0],
							[0, 0, 1]])
	

	roll_matrix = ti.Matrix([
		[1, 0, 0],
		[0, tm.cos(angle_pitch), -tm.sin(angle_pitch)],
		[0, tm.sin(angle_pitch), tm.cos(angle_pitch)]
	])
	for i, j in pixels:
		u = i / (image_width - 1)
		v = j / (image_height - 1)
		d =  lower_left_corner + u * horizontal + v * vertical
		d.x = d.x
		dy = d.y
		d.y = d.z
		d.z = dy
		d = roll_matrix @ d 
		d = yaw_matrix @ d
		
		d.x = d.x
		dy = d.y
		d.y = d.z
		d.z = dy
		
		
		d += offset
		gbuff(o=o,d=d,i=i,j=j)
		dif = vec3(0)
		specdif =vec3(0)
		for sa in range(2):
			dif += render_diffuse(o,d,i,j) 
		diffuse[i,j] =  (dif / 2)
		c = vec3(0)
		if (normals[i,j].x == 0 and normals[i,j].y == 0 and normals[i,j].z == 0):
			c = ray_color(ray_direction=d,ray_origin=o)
			specular_buff[i,j] = vec3(0)
		
		sky[i,j] = c

	for i, j in pixels:
		pixels[i,j] = (albedo[i,j] * diffuse[i,j]*ref[i,j])+(diffuse[i,j]*(1-ref[i,j])) + sky[i,j]
	for i, j in pixels:
		pixels[i,j] = ACESFilm(pixels[i,j])
		pixels[i,j] = (buff[i,j] + pixels[i,j])/2
		buff[i,j] = pixels[i,j]
	for i, j in acc_buff:	
		if(acc == 1):
			acc_buff[i,j] = acc_buff[i,j] + pixels[i,j]
			pixels[i,j] = acc_buff[i,j] / w
		else:
			acc_buff[i,j] = pixels[i,j]
	
		
		
		
gui = ti.GUI("render", res=(image_width, image_height))
diffuse_gui = ti.GUI("diffuse", res=(image_width, image_height),fast_gui=True)

w = 1

import requests
import time
import json

print(xmax)
print(ymax)
print(zmax)
rot = vec3(0)
last_origin = vec3(0,0,0)


offx = 5680
offy = 63
offz = 7829


r = requests.get("http://127.0.0.1:2008/chunk?x=5680&y=63&z=7829")
blocks = json.loads(r.text)
blocks = blocks["blocks"]


r = requests.get("http://127.0.0.1:2008/chunk?x=5680&y=73&z=7829")
blocks2 = json.loads(r.text)
blocks2 = blocks2["blocks"]

blocks = blocks + blocks2 

for i, b in enumerate(blocks):
	blocks[i] = b[33:-11]

ids = ["air"]
print(len(blocks))
count = 0
for i in range(16):
	for j in range(16):
		for k in range(16):
			if blocks[count] == "air":
				pass
			else:
				if blocks[count] in ids:
					chunk[i,j,k] = ids.index(blocks[count])
				else:
					ids.append(blocks[count])
					chunk[i,j,k] = ids.index(blocks[count])
					try:
						img = im.open(("block\\" + blocks[count] + ".png"))
						mer = im.open(("block\\" + blocks[count] + "_s.png"))
						print("open")
					except:
						try:
							img = im.open(("block\\" + blocks[count] + "_top.png"))
							mer = im.open(("block\\" + blocks[count] + "_s_top.png"))
						except:
							img = im.open("block\\stone.png")
							mer = im.open("block\\stone_s.png")
					rgb_im = img.convert('RGBA')
					mer_im = mer.convert('RGBA')

					for x in range(16):
						for y in range(16):
							r, g, b , a = rgb_im.getpixel((x, y))
							texture[x + (chunk[i,j,k] * 16) ,y,0] = tm.vec4(r,g,b,1) / 255
							r, g, b, e = mer_im.getpixel((x, y))
							texture[x + (chunk[i,j,k] * 16) ,y,1] = tm.vec4(r,g,b,e) / 255
							if(e == 255):
								texture[x + (chunk[i,j,k] * 16) ,y,2] = tm.vec4(1)
							else:
								texture[x + (chunk[i,j,k] * 16) ,y,2] = tm.vec4(1+((e/244)*10))
							
			count += 1
			
			
			
acc = 0
while gui.running:
	vr = requests.get(url="http://127.0.0.1:2008/data")
	vr.close()
	data = json.loads(vr.text)  
	origin.x = float(data["x"]) - offx - 0.5
	origin.y = float(data["y"]) + 1.63 -offy - 0.5
	origin.z = float(data["z"]) - offz - 0.5
	if(last_origin.x == origin.x and last_origin.y == origin.y and last_origin.z == origin.z):
		acc = 1
	else:
		acc = 0
		w = 1
	rot.x = (float(data["yaw"])) * (3.14159 / 180)  + (180 * (3.14159 / 180))
	rot.y = (float(data["pitch"]))  * (3.14159 / 180)
	rot.z = (float(data["roll"]))
	paint(origin,w=w,rot=rot,last_origin=last_origin, acc=acc)
	gui.set_image(test)
	gui.show()
	diffuse_gui.set_image(pixels)
	diffuse_gui.show()
	w += 1  
	last_origin.x = float(data["x"]) - offx - 0.5
	last_origin.y = float(data["y"]) + 1.63 - offy - 0.5
	last_origin.z = float(data["z"]) - offz - 0.5