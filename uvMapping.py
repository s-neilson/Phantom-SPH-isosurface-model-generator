import math
import numpy
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

from halfedge import pointInsideTriangle


def determineVertexColour(vertexValue,valueMinimum,valueMaximum):
    clampedVertexValue=max(min(vertexValue,valueMaximum),valueMinimum) #vertexValue is clamped to the minimum and maximum values.
    hue=(0.75*(clampedVertexValue-valueMinimum))/(valueMaximum-valueMinimum)
    vertexColourRgb=hsv_to_rgb([hue,1.0,1.0]) #RGB colour hue depends on vertexValue.
    return vertexColourRgb

#Determines the colours of a set of UV vertices and outputs lists that can be used to interpolate the colours across UV triangles.
def determineUvVertexColours(meshToMap,vertexValues,valueMinimum,valueMaximum):
    for currentVertex,currentVertexValue in zip(meshToMap.vertices,vertexValues):
        for currentUvVertex in currentVertex.uvVertices.values(): #Loops over each UV vertex associated with each real vertex.
            r,g,b=determineVertexColour(currentVertexValue,valueMinimum,valueMaximum)
            currentUvVertex.setColour(r,g,b)
    


#Performs an equirectangular projection on a set of vertices and outputs a set of UV coordinates so a texture can be used. Also returns the angular width of the map in radians.
def determineEquirectangularUvPositions(originPosition,meshToMap):
    for currentVertex in meshToMap.vertices:
        currentVertexPosition=currentVertex.position
        currentVertexFaces=currentVertex.getFaces()

        r=currentVertexPosition-originPosition
        rDistance=numpy.linalg.norm(r) #Total distance to the vertex from the origin.
        phi=math.asin(r[2]/rDistance) #Vertical angle.
        theta=math.atan2(r[0],(-1.0)*r[1]) #Clockwise angle in XY plane from negative Y axis.
            
        #Phi is set so 0 is the negative Z axis and pi is the positive Z axis. Theta is scaled from 0 to 2pi, with the 0 and 2pi position being the positive Y axis.
        phi+=((math.pi)/2.0)
        theta+=math.pi
        meshToMap.addUvVertex(theta,phi,currentVertex,currentVertexFaces)
        

        
    maximumThetaAngle=2.0*(math.pi) #The maxmimum theta angle taking the duplicated vertices into account. Is used to scale the horizintal UV coordinates.
    
    #Checks are done to determine the faces that have edges that cross the 0 and 2pi theta boundary. Duplicate UV vertex entries need to be made for them
    #to prevent the faces from wrapping around to the opposite side of the texture.
    vertexBoundaryCrossings={} #A dictionary with the vertices responsible for boundary crossings as keys and the effected faces as values. Also contains the phi and updated theta value for the duplicated UV vertices.
    for currentFace in meshToMap.faces:
        currentFaceVertices1=currentFace.getAllVertices()
        currentFaceVertices2=[currentHalfedge.oppositeHalfedge.endVertex for currentHalfedge in currentFace.getAllHalfedges()]
        
        for v1,v2 in zip(currentFaceVertices1,currentFaceVertices2): #Loops over all edges of the face.
            v1x,v1y=v1.position[0],v1.position[1]
            v2x,v2y=v2.position[0],v2.position[1]
            
            #Checks which vertex (if any) has caused the edge to cross from negative x to positive x in the positive y half of the model.
            v1cb=(v1x<originPosition[0]) and (v1y>originPosition[1]) and (v2x>=originPosition[0]) and (v2y>=originPosition[1])
            v2cb=(v2x<originPosition[0]) and (v2y>originPosition[1]) and (v1x>=originPosition[0]) and (v1y>=originPosition[1])
            
            if(v1cb): #Notes that v1 is a boundary crossing vertex and assigns the current face as one that is affected.
                if(v1 not in vertexBoundaryCrossings):
                    v1Theta,v1Phi=list(v1.uvVertices.values())[0].position[0],list(v1.uvVertices.values())[0].position[1] #The theta value of the UV vertex already assigned to v1.
                    v1Theta+=(2.0*(math.pi)) #The duplicate vertex theta angle has 2pi added to it, meaning it is on the right hand side of the texture.
                    maximumThetaAngle=max(maximumThetaAngle,v1Theta) #The maxmimum theta angle is updated if necessary.
                    vertexBoundaryCrossings[v1]=(v1Theta,v1Phi,[]) #An empty list is created for the current vertex if it has not been added as a key to the dictionary yet.
                    
                vertexBoundaryCrossings[v1][2].append(currentFace)
                
            if(v2cb):
                if(v2 not in vertexBoundaryCrossings):
                    v2Theta,v2Phi=list(v2.uvVertices.values())[0].position[0],list(v2.uvVertices.values())[0].position[1] 
                    v2Theta+=(2.0*(math.pi))
                    maximumThetaAngle=max(maximumThetaAngle,v2Theta)
                    vertexBoundaryCrossings[v2]=(v2Theta,v2Phi,[])
                    
                vertexBoundaryCrossings[v2][2].append(currentFace)

        
    #New UV vertices are created and assigned to the correct faces for the vertices that cross the boundary.
    for currentVertex,(theta,phi,faces) in zip(vertexBoundaryCrossings.keys(),vertexBoundaryCrossings.values()):
        meshToMap.addUvVertex(theta,phi,currentVertex,faces)
            
                 
    #The angles are scaled between 0 and their maximum values to produce UV coordinates instead.
    for currentUvVertex in meshToMap.uvVertices:
        currentUvVertex.position[0]/=maximumThetaAngle
        currentUvVertex.position[1]/=(math.pi)
        
    return maximumThetaAngle
                  

    
#Fills the pixels inside a triangle on a texture with colours interpolated from the triangle's vertices.
def interpolateTriangleColours(texture,v1r,v1g,v1b,v1z0,v2r,v2g,v2b,v2z0,v3r,v3g,v3b,v3z0):
    textureWidth=texture.shape[1]
    textureHeight=texture.shape[0]

    #If the UV coordinates along with a colour are considered points in 3d space, planes (one for red, green and blue) can be fitted to the triangle's vertices in order to interpolate
    #the colour at any point inside the triangle. Below are the normal vectors for these planes.
    nR=numpy.cross(numpy.subtract(v1r,v2r),numpy.subtract(v3r,v2r))
    nR/=numpy.linalg.norm(nR)
    nR=nR.tolist()
    nG=numpy.cross(numpy.subtract(v1g,v2g),numpy.subtract(v3g,v2g))
    nG/=numpy.linalg.norm(nG)
    nG=nG.tolist()
    nB=numpy.cross(numpy.subtract(v1b,v2b),numpy.subtract(v3b,v2b))
    nB/=numpy.linalg.norm(nB) 
    nB=nB.tolist()
        
    
    #A search is done within the bounding box of the triangle for points that are inside the triangle.
    uMinimum=max(0,math.floor(min(v1z0[0],v2z0[0],v3z0[0])))
    uMaximum=min(textureWidth,math.floor(max(v1z0[0],v2z0[0],v3z0[0]))+1)
    vMinimum=max(0,math.floor(min(v1z0[1],v2z0[1],v3z0[1])))
    vMaximum=min(textureHeight,math.floor(max(v1z0[1],v2z0[1],v3z0[1]))+1)
    
    for uI in range(uMinimum,uMaximum):
        for vI in range(vMinimum,vMaximum):
            u=uI+0.5 #The centre of each pixel in the bounding box is evaluated.
            v=vI+0.5
            
            if(pointInsideTriangle((v1z0[0],v1z0[1]),(v2z0[0],v2z0[1]),(v3z0[0],v3z0[1]),(u,v))):
                interpolatedR_u=nR[0]*u
                interpolatedR_v=nR[1]*v
                interpolatedR=(numpy.dot(nR,v1r)-interpolatedR_u-interpolatedR_v)/nR[2] #Uses the point normal equation of a plane to calculate the colour channel value.
                
                interpolatedG_u=nG[0]*u
                interpolatedG_v=nG[1]*v
                interpolatedG=(numpy.dot(nG,v1g)-interpolatedG_u-interpolatedG_v)/nG[2]
                
                interpolatedB_u=nB[0]*u
                interpolatedB_v=nB[1]*v
                interpolatedB=(numpy.dot(nB,v1b)-interpolatedB_u-interpolatedB_v)/nB[2]

                texture[vI,uI,:]=[interpolatedR,interpolatedG,interpolatedB] #The colour of the current pixel in the texture is set.
            
                

        
    
#Creates a texture by filling each triangle of a UV map with colours interpolated from the triangle vertices.
def fillTexture(meshToMap,textureWidth,textureHeight,textureName):
    texture=numpy.zeros(shape=(textureHeight,textureWidth,3))
    uvScalingFactor=numpy.array([textureWidth,textureHeight,1.0])
    
    for currentTriangle in meshToMap.faces: #Loops over all triangles.
        uv1,uv2,uv3=currentTriangle.getAllUvVertices()
        
        v1r,v1g,v1b,v1z0=uv1.r*uvScalingFactor,uv1.g*uvScalingFactor,uv1.b*uvScalingFactor,uv1.position*uvScalingFactor
        v2r,v2g,v2b,v2z0=uv2.r*uvScalingFactor,uv2.g*uvScalingFactor,uv2.b*uvScalingFactor,uv2.position*uvScalingFactor
        v3r,v3g,v3b,v3z0=uv3.r*uvScalingFactor,uv3.g*uvScalingFactor,uv3.b*uvScalingFactor,uv3.position*uvScalingFactor
        
        interpolateTriangleColours(texture,v1r,v1g,v1b,v1z0,v2r,v2g,v2b,v2z0,v3r,v3g,v3b,v3z0)
        
    plt.imsave(fname=textureName+".png",arr=texture,origin="lower",dpi=1.0)
    
    