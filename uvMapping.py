import copy
import math
import numpy
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt


def determineVertexColour(vertexValue,valueMinimum,valueMaximum):
    clampedVertexValue=max(min(vertexValue,valueMaximum),valueMinimum) #vertexValue is clamped to the minimum and maximum values.
    hue=(0.75*(clampedVertexValue-valueMinimum))/(valueMaximum-valueMinimum)
    vertexColourRgb=hsv_to_rgb([hue,1.0,1.0]) #RGB colour hue depends on vertexValue.
    return vertexColourRgb


#Performs an equirectangular projection on a set of vertices and outputs a set of UV coordinates so a texture can be used.
#It also returns the vertex colours for each of the vertex UV entries based on a supplied value for each vertex.
def determineEquirectangularUvPositions(originPosition,vertexPositions,faces,vertexValues,valueMinimum,valueMaximum):
    vertexAngularPositions=[] #A list of the angular coordinates of the vertices in the mesh.

    for currentVertexPosition in vertexPositions:
        r=numpy.linalg.norm(currentVertexPosition-originPosition) #Total distance to the vertex from the origin.
        phi=math.asin(currentVertexPosition[2]/r) #Vertical angle.
        theta=math.atan2(currentVertexPosition[0],(-1.0)*currentVertexPosition[1]) #Clockwise angle in XY plane from negative Y axis.
            
        #Phi is set so 0 is the negative Z axis and pi is the positive Z axis. Theta is scaled from 0 to 2pi, with the 0 and 2pi position being the positive Y axis.
        phi+=((math.pi)/2.0)
        theta+=math.pi
        vertexAngularPositions.append([theta,phi])
        

        
     
    vertexUvColours=[determineVertexColour(vertexValues[i,0],valueMinimum,valueMaximum) for i in range(0,vertexValues.shape[0])] #The colours of each vertex in the UV map. These colours will be interpolated over the triangles.
    facesUvIndices=copy.deepcopy(faces) #A list that contains the UV entry indices for the faces. Is initally the same as the input face list.
    maximumThetaAngle=2.0*(math.pi) #The maxmimum theta angle taking the duplicated vertices into account. Is used to scale the horizintal UV coordinates.
    
    #Checks are done to determine the faces that have edges that cross the 0 and 2pi theta boundary. Duplicate UV vertex entries need to be made for them
    #to prevent the faces from wrapping around to the opposite side of the texture.
    for currentFace in facesUvIndices:
        vertex1Index=currentFace[0]-1
        vertex2Index=currentFace[1]-1
        vertex3Index=currentFace[2]-1
        vertex1x=vertexPositions[vertex1Index][0]
        vertex2x=vertexPositions[vertex2Index][0]
        vertex3x=vertexPositions[vertex3Index][0]
        vertex1y=vertexPositions[vertex1Index][1]
        vertex2y=vertexPositions[vertex2Index][1]
        vertex3y=vertexPositions[vertex3Index][1]
        
        #Checks if various edges for the current face have vertices across the positive y axis.
        v1cb=(vertex1x<0.0) and (vertex1y>0.0) and (((vertex2x>=0.0) and (vertex2y>0.0)) or ((vertex3x>=0.0) and (vertex3y>0.0))) #Vertex 1 has crossed the boundary.
        v2cb=(vertex2x<0.0) and (vertex2y>0.0) and (((vertex1x>=0.0) and (vertex1y>0.0)) or ((vertex3x>=0.0) and (vertex3y>0.0))) #Vertex 2 has crossed the boundary.
        v3cb=(vertex3x<0.0) and (vertex3y>0.0) and (((vertex1x>=0.0) and (vertex1y>0.0)) or ((vertex2x>=0.0) and (vertex2y>0.0))) #Vertex 3 has crossed the boundary.

        
        if(v1cb):
            currentAngularPosition=copy.deepcopy(vertexAngularPositions[vertex1Index])
            currentAngularPosition[0]+=(2.0*(math.pi)) #The duplicate vertex theta angle has 2pi added to it, meaning it is on the right hand side of the texture.
            maximumThetaAngle=max(maximumThetaAngle,currentAngularPosition[0]) #The maxmimum theta angle is updated if necessary.
            vertexAngularPositions.append(currentAngularPosition)
            
            currentUvEntryIndex=len(vertexAngularPositions) #Is the index of the previously duplicated vertex UV entry as it was the last one added.
            currentFace[0]=currentUvEntryIndex #The index for the vertex that is responsible for an edge being over the boundary is updated.
            
            currentVertexUvColour=vertexUvColours[vertex1Index] #The vertex colour for the original vertex is duplicated.
            vertexUvColours.append(currentVertexUvColour)
            
        if(v2cb):
            currentAngularPosition=copy.deepcopy(vertexAngularPositions[vertex2Index])
            currentAngularPosition[0]+=(2.0*(math.pi))
            maximumThetaAngle=max(maximumThetaAngle,currentAngularPosition[0])
            vertexAngularPositions.append(currentAngularPosition)
            
            currentUvEntryIndex=len(vertexAngularPositions)
            currentFace[1]=currentUvEntryIndex
            
            currentVertexUvColour=vertexUvColours[vertex2Index]
            vertexUvColours.append(currentVertexUvColour)
            
        if(v3cb):
            currentAngularPosition=copy.deepcopy(vertexAngularPositions[vertex3Index])
            currentAngularPosition[0]+=(2.0*(math.pi))
            maximumThetaAngle=max(maximumThetaAngle,currentAngularPosition[0])
            vertexAngularPositions.append(currentAngularPosition)
            
            currentUvEntryIndex=len(vertexAngularPositions)
            currentFace[2]=currentUvEntryIndex
            
            currentVertexUvColour=vertexUvColours[vertex3Index]
            vertexUvColours.append(currentVertexUvColour)
            
      
    #The outputed UV coordinates are scaled to between 0 and 1. UV vertex lists are given for all three colour channels and a forth one where the third coordinate is equal to zero.
    vertexUvR=[]
    vertexUvG=[]
    vertexUvB=[]
    vertexUvZ0=[]
    
    for currentAngularPosition,currentVertexColour in zip(vertexAngularPositions,vertexUvColours):
        u=currentAngularPosition[0]/maximumThetaAngle
        v=currentAngularPosition[1]/(math.pi)
        
        vertexUvR.append(numpy.array([u,v,currentVertexColour[0]]))
        vertexUvG.append(numpy.array([u,v,currentVertexColour[1]]))
        vertexUvB.append(numpy.array([u,v,currentVertexColour[2]]))
        vertexUvZ0.append(numpy.array([u,v,0.0]))
               
    return vertexUvR,vertexUvG,vertexUvB,vertexUvZ0,facesUvIndices



    
#Fills the pixels inside a triangle on a texture with colours interpolated from the triangle's vertices.
def interpolateTriangleColours(texture,v1r,v1g,v1b,v1z0,v2r,v2g,v2b,v2z0,v3r,v3g,v3b,v3z0):
    textureWidth=texture.shape[1]
    textureHeight=texture.shape[0]
    
    #These are the 2d normal vectors for the edges of the UV map triangle (pointing outwards assuming that the vertices are ordered in an anti-clockwise order).
    nE31=numpy.cross(numpy.subtract(v1z0,v3z0),[0.0,0.0,1.0])
    nE31/=numpy.linalg.norm(nE31)
    nE31=nE31.tolist()
    nE23=numpy.cross(numpy.subtract(v3z0,v2z0),[0.0,0.0,1.0])
    nE23/=numpy.linalg.norm(nE23)
    nE23=nE23.tolist()
    nE12=numpy.cross(numpy.subtract(v2z0,v1z0),[0.0,0.0,1.0])
    nE12/=numpy.linalg.norm(nE12)
    nE12=nE12.tolist()
       
    
    
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
        
    
    #A search is done within the bounidng box of the triangle for points that are inside the triangle.
    uMinimum=max(0,math.floor(min(v1z0[0],v2z0[0],v3z0[0])))
    uMaximum=min(textureWidth,math.floor(max(v1z0[0],v2z0[0],v3z0[0]))+1)
    vMinimum=max(0,math.floor(min(v1z0[1],v2z0[1],v3z0[1])))
    vMaximum=min(textureHeight,math.floor(max(v1z0[1],v2z0[1],v3z0[1]))+1)
    
    for uI in range(uMinimum,uMaximum):
        for vI in range(vMinimum,vMaximum):
            u=uI+0.5 #The centre of each pixel is evaluated.
            v=vI+0.5
            
            #The expressions below determine if the current pixel is inside a triangle edge by checking if the perpendicular distance to it is less than or equal to zero.
            #The perpendicular distance is calculated by taking the dot product of the edge normal vector with the vector pointing from one of the edge's vertices to the current pixel's centre.
            insideE31=((nE31[0]*(u-v1z0[0]))+(nE31[1]*(v-v1z0[1])))<=0.0
            insideE23=((nE23[0]*(u-v3z0[0]))+(nE23[1]*(v-v3z0[1])))<=0.0
            insideE12=((nE12[0]*(u-v2z0[0]))+(nE12[1]*(v-v2z0[1])))<=0.0

            if(insideE31 and insideE23 and insideE12): #If the point is inside the triangle due to it being within all three edges.
                interpolatedR_u=nR[0]*(u-v1r[0])
                interpolatedR_v=nR[1]*(v-v1r[1])
                interpolatedR=v1r[2]+(((-1.0)*(interpolatedR_u+interpolatedR_v))/nR[2]) #Uses the point normal equation of a plane to calculate the colour channel value.
                
                interpolatedG_u=nG[0]*(u-v1g[0])
                interpolatedG_v=nG[1]*(v-v1g[1])
                interpolatedG=v1g[2]+(((-1.0)*(interpolatedG_u+interpolatedG_v))/nG[2])
                
                interpolatedB_u=nB[0]*(u-v1b[0])
                interpolatedB_v=nB[1]*(v-v1b[1])
                interpolatedB=v1b[2]+(((-1.0)*(interpolatedB_u+interpolatedB_v))/nB[2])

                #The colour of the current pixel in the texture is set.
                texture[vI,uI,:]=[interpolatedR,interpolatedG,interpolatedB]

            
    
    
    
#Creates a texture by filling each triangle of a UV map with colours interpolated from the triangle vertices.
def fillTexture(vertexUvR,vertexUvG,vertexUvB,vertexUvZ0,triangles,textureHeight,textureName):
    textureWidth=math.floor(2.25*textureHeight) #The texture width represents 360+45 (405) degrees while the height represents 180 degrees.
    texture=numpy.zeros(shape=(textureHeight,textureWidth,3))
    uvScalingFactor=numpy.array([textureWidth,textureHeight,1.0])
    
    for currentTriangle in triangles: #Loops over all triangles.
        vertex1Index=currentTriangle[0]-1
        vertex2Index=currentTriangle[1]-1
        vertex3Index=currentTriangle[2]-1
        
        v1r=vertexUvR[vertex1Index]*uvScalingFactor
        v1g=vertexUvG[vertex1Index]*uvScalingFactor
        v1b=vertexUvB[vertex1Index]*uvScalingFactor
        v1z0=vertexUvZ0[vertex1Index]*uvScalingFactor
        
        v2r=vertexUvR[vertex2Index]*uvScalingFactor
        v2g=vertexUvG[vertex2Index]*uvScalingFactor
        v2b=vertexUvB[vertex2Index]*uvScalingFactor
        v2z0=vertexUvZ0[vertex2Index]*uvScalingFactor
        
        v3r=vertexUvR[vertex3Index]*uvScalingFactor
        v3g=vertexUvG[vertex3Index]*uvScalingFactor
        v3b=vertexUvB[vertex3Index]*uvScalingFactor
        v3z0=vertexUvZ0[vertex3Index]*uvScalingFactor
        
        interpolateTriangleColours(texture,v1r.tolist(),v1g.tolist(),v1b.tolist(),v1z0.tolist(),v2r.tolist(),v2g.tolist(),v2b.tolist(),v2z0.tolist(),v3r.tolist(),v3g.tolist(),v3b.tolist(),v3z0.tolist())
        
    textureFigure=plt.figure(figsize=(2.25,1.0),dpi=float(textureHeight))
    textureAxes=textureFigure.gca()
    textureAxes.set_axis_off()
    textureAxes.imshow(texture,origin="lower")
    textureFigure.savefig(textureName+".png",dpi="figure",bbox_inches="tight",pad_inches=0.0)

    
    
    
    
    
    
    