import xml.etree.ElementTree as ET
import numpy as np, copy
import xml.dom.minidom as MD

ROLL_DEG=0.0
def rpy_to_mat(r,p,y):
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]); Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]); Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz@Ry@Rx
def mat_to_rpy(R):
    p=np.arctan2(-R[2,0],np.sqrt(R[0,0]**2+R[1,0]**2)); y=np.arctan2(R[1,0],R[0,0]); r=np.arctan2(R[2,1],R[2,2])
    return np.array([r,p,y])
# RUKA fingers point along -Z in base frame now (z=-0.14). G1 hand extends +X.
# Need rotate -Z -> +X: Ry(-90): [0,0,-1]->? Ry(t)@[0,0,-1]=[-sin t,0,-cos t]; for [1,0,0] need t=-90
Rm = rpy_to_mat(np.deg2rad(ROLL_DEG),0,0) @ rpy_to_mat(0,-np.pi/2,0)
rpy_attach=mat_to_rpy(Rm)
XYZ=[0.0415,0.003,0.0]
print("attach rpy(deg):",np.rad2deg(rpy_attach))

g1=ET.parse('/mnt/user-data/uploads/g1_29dof_rev_1_0.urdf'); groot=g1.getroot()
for j in groot.findall('joint'):
    if j.get('name')=='left_hand_palm_joint': groot.remove(j)
for l in groot.findall('link'):
    if l.get('name')=='left_rubber_hand': groot.remove(l)

ruka=ET.parse('/home/claude/ruka_left_fixed.urdf'); rroot=ruka.getroot()
P='ruka_'
def nn(n): return P+n
for l in rroot.findall('link'):
    nl=copy.deepcopy(l); nl.set('name',nn(l.get('name')))
    for m in nl.iter('mesh'): m.set('filename','meshes/'+m.get('filename').split('/')[-1])
    groot.append(nl)
for j in rroot.findall('joint'):
    nj=copy.deepcopy(j); nj.set('name',nn(j.get('name')))
    nj.find('parent').set('link',nn(nj.find('parent').get('link')))
    nj.find('child').set('link',nn(nj.find('child').get('link')))
    groot.append(nj)
aj=ET.SubElement(groot,'joint'); aj.set('name','left_wrist_to_ruka'); aj.set('type','fixed'); aj.set('dont_collapse','true')
o=ET.SubElement(aj,'origin'); o.set('xyz',f"{XYZ[0]} {XYZ[1]} {XYZ[2]}"); o.set('rpy',f"{rpy_attach[0]:.8f} {rpy_attach[1]:.8f} {rpy_attach[2]:.8f}")
ET.SubElement(aj,'parent').set('link','left_wrist_yaw_link'); ET.SubElement(aj,'child').set('link',nn('base_link'))
groot.set('name','g1_29dof_left_arm_ruka')
xml='\n'.join(l for l in MD.parseString(ET.tostring(groot)).toprettyxml(indent='  ').split('\n') if l.strip())
open('/home/claude/combined2/g1_left_arm_ruka.urdf','w').write(xml)
print(f"WROTE combined2/g1_left_arm_ruka.urdf  links={len(groot.findall('link'))} joints={len(groot.findall('joint'))}")
