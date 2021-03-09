import h5py
import math
import numpy
from sklearn.neighbors import KDTree

from interpolatedValues import interpolateGrid_density
from interpolatedValues import interpolateList_vNormalXY
from interpolatedValues import interpolateList_vNormalR

from dualContouring import processEdgeIntersections
from dualContouring import determineVertices
from dualContouring import createMeshDataLists

from uvMapping import determineEquirectangularUvPositions
from uvMapping import fillTexture



#Creates an obj file based on mesh data.
def writeObjFile(filename,objectList):
    objFile=open(filename+".obj","wt")
    vertexPositionIndexOffset=0 #Offsets the vertex position and normal indices for each object meaning that the face vertex indices are relative to the object they are in.
    vertexUvIndexOffset=0 #Similar to vertexPositionIndexOffset but for the vertex UV entries.
    
    for currentObject in objectList: #Loops through all sets of mesh objects.
        objFile.write("o "+currentObject[0]+"\n") #The mesh object name.
        
        for currentVertexPosition in currentObject[1]: #Vertex positions.
            positionString="v "+str(currentVertexPosition[0])+" "+str(currentVertexPosition[1])+" "+str(currentVertexPosition[2])+"\n"
            objFile.write(positionString)
            
        if((currentObject[2] is not None) and (currentObject[5] is not None)): #If the mesh has a UV map.
            for currentVertexUV in currentObject[2]: #UV positions for the vertices.
                uvString="vt "+str(currentVertexUV[0])+" "+str(currentVertexUV[1])+"\n"
                objFile.write(uvString)
        
            
        for currentVertexNormal in currentObject[3]: #Vertex normals.
            normalString="vn "+str(currentVertexNormal[0])+" "+str(currentVertexNormal[1])+" "+str(currentVertexNormal[2])+"\n"
            objFile.write(normalString)
                       
        
        objFile.write("s on\n") #Smooth shading is enabled; normals across the face are interpolated using the normals of the face's vertices.
        for i,currentFace in enumerate(currentObject[4]): #Faces.
            vertex1Index=currentFace[0]+vertexPositionIndexOffset
            vertex2Index=currentFace[1]+vertexPositionIndexOffset
            vertex3Index=currentFace[2]+vertexPositionIndexOffset
            
            faceString=""
            if((currentObject[2] is not None) and (currentObject[5] is not None)): #If there is a UV map.
                vertex1UvIndex=currentObject[5][i][0]+vertexUvIndexOffset
                vertex2UvIndex=currentObject[5][i][1]+vertexUvIndexOffset
                vertex3UvIndex=currentObject[5][i][2]+vertexUvIndexOffset
                
                faceString="f "+str(vertex1Index)+"/"+str(vertex1UvIndex)+"/"+str(vertex1Index)+" "+str(vertex2Index)+"/"+str(vertex2UvIndex)+"/"+str(vertex2Index)+" "+str(vertex3Index)+"/"+str(vertex3UvIndex)+"/"+str(vertex3Index)+"\n"
            else: #If there is no UV map.
                faceString="f "+str(vertex1Index)+"//"+str(vertex1Index)+" "+str(vertex2Index)+"//"+str(vertex2Index)+" "+str(vertex3Index)+"//"+str(vertex3Index)+"\n"
            objFile.write(faceString)
            
        vertexPositionIndexOffset+=len(currentObject[1]) #The vertex indices are offset for the next object by the total number of vertices in the current object.
        vertexUvIndexOffset+=len(currentObject[2]) if((currentObject[2] is not None) and (currentObject[5] is not None)) else 0 #Updated in a similar way to vertexPositionIndexOffset.
           
    objFile.close()
    
    
    
#Creates the mesh data for a sphere.    
def createSphereMeshData(centre,radius,zDivisions,xyDivisions):
    vertexPositions=[]
    vertexNormals=[]
    faces=[]
    
    #The vertex positions and normals are added.
    for phi in numpy.linspace(0.5*math.pi,(-0.5)*math.pi,zDivisions): #Vertical angle.
        if(phi==(0.5*math.pi)): #North pole.
            vertexPositions.append(centre+numpy.array([0.0,0.0,radius]))
            vertexNormals.append(numpy.array([0.0,0.0,1.0]))
            continue
        
        if(phi==((-0.5)*math.pi)): #South pole.
            vertexPositions.append(centre+numpy.array([0.0,0.0,(-1.0)*radius]))
            vertexNormals.append(numpy.array([0.0,0.0,-1.0]))
            continue
        
        z=centre[2]+(radius*math.sin(phi))
        currentVertexRingRadius=radius*math.cos(phi) #Radius of vertex ring at the current height from the sphere's centre.
               
        for theta in numpy.linspace(0.0,2.0*math.pi,xyDivisions,endpoint=False): #Horizontal angle.
            x=centre[0]+(currentVertexRingRadius*math.cos(theta))
            y=centre[1]+(currentVertexRingRadius*math.sin(theta))
            
            currentVertexPosition=numpy.array([x,y,z])
            vertexPositions.append(currentVertexPosition)
            vertexNormals.append((currentVertexPosition-centre)/radius)
            
            
    #The faces are added.
    for phiIndex in range(1,zDivisions):
        for thetaIndex in range(0,xyDivisions):
            #The indices of vertices to make up the current face/s to be added.
            bottomRightIndex=2+thetaIndex+(xyDivisions*(phiIndex-1))
            bottomLeftIndex=bottomRightIndex+(xyDivisions-1) if(thetaIndex==0) else bottomRightIndex-1 #At the start of a vertex ring the vertex index immediently to the left was the last one added in that ring.
            topRightIndex=bottomRightIndex-xyDivisions #Non pole vertices immediately above or below a non pole vertex are xyDivisions positions away.  
            topLeftIndex=bottomLeftIndex-xyDivisions
            
            if(phiIndex==1): #The first vertex ring below the north pole.
                faces.append([bottomLeftIndex,bottomRightIndex,1]) #A single triangle is added attached to the north pole.
            elif(phiIndex==(zDivisions-1)): #The south pole.
                faces.append([(xyDivisions*(zDivisions-2))+2,topRightIndex,topLeftIndex]) #A single triangle is added attached to the south pole.
            else: #When another vertex ring is above the current one.
                faces.append([bottomLeftIndex,bottomRightIndex,topRightIndex])
                faces.append([topRightIndex,topLeftIndex,bottomLeftIndex])
                
    return vertexPositions,vertexNormals,faces
            
            
                        
            
#Creates a grid of coordinates where the densities and density gradients will be sampled.
def createSamplingGridPositions(cubeAxisCount,cubeSize):
    outputGrid=[[[None for z in range(0,cubeAxisCount+1)] for y in range(0,cubeAxisCount+1)] for x in range(0,cubeAxisCount+1)]
    
    cornerPosition=(float(cubeAxisCount)*cubeSize)/(2.0) #The grid is centred at the position 0,0,0.
    axesOrdinates=numpy.linspace((-1.0*cornerPosition),cornerPosition,cubeAxisCount+1)
    axesXarray,axesYarray,axesZarray=numpy.meshgrid(axesOrdinates,axesOrdinates,axesOrdinates,indexing="ij")
    
    #Loops over all grid cubes
    for x in range(0,cubeAxisCount+1):
        for y in range(0,cubeAxisCount+1):
            for z in range(0,cubeAxisCount+1):
                outputGrid[x][y][z]=numpy.array([axesXarray[x,y,z],axesYarray[x,y,z],axesZarray[x,y,z]])
                
    return outputGrid



                
                
inputFilename="trinary_07630"  
outputFilename="twoPlanet19Years"
cubeAxisCount=100 #Number of cubes to split the area up in all three axes. Number of sample points in an axis is equal to this number plus 1.
cubeSize=10.0 #Width of cube side length
textureHeight=1000 #Does not correspond exactly to height in pixels.
isosurfaceDensityThreshold=5.1e-10 #The isosurface represents this density. In units of solar masses/solar radii^3
#Below are the minimum and maximum velocities the textures will represent.
vXYmin=-0.01
vXYmax=0.01
vRmin=-0.01
vRmax=0.01


 
loadedFile=h5py.File(inputFilename+".h5","r")
particleMass=loadedFile["header"]["massoftype"][0] #The mass of each SPH particle.
particleCount=loadedFile["header"]["nparttot"][()] #The number of SPH particles.
particleVelocities=numpy.array(loadedFile["particles"]["vxyz"])
sinkPositions=loadedFile["sinks"]["xyz"]

particleH=numpy.array(loadedFile["particles"]["h"])
particlePositions=numpy.array(loadedFile["particles"]["xyz"])
particleTree=KDTree(particlePositions) #Stores particle positions in a kd tree so the neighbouring particles to a sampling location can be quickly found.



print("Creating sampling grid.")
samplingGridPositions=createSamplingGridPositions(cubeAxisCount,cubeSize)
densities=interpolateGrid_density(samplingGridPositions,particleTree,particleMass,particleH,particlePositions)
print("Finding grid-isosurface intersections.")
edgeIntersections,edgeIntersectionNormals,edgeNeighbouringCubes,cubeEdgeIntersections=processEdgeIntersections(5.1e-10,True,densities,samplingGridPositions,cubeAxisCount,
                                                                                                                particleTree,particleMass,particleH,particlePositions)
print("Placing vertices.")
cubeVertices=determineVertices(True,cubeEdgeIntersections,samplingGridPositions,cubeAxisCount,cubeSize,
                               edgeIntersections,edgeIntersectionNormals,
                               particleTree,particleMass,particleH,particlePositions)

print("Triangulating isosurface mesh.")
starMeshVertices,starMeshVertexNormals,starMeshFaces=createMeshDataLists(cubeAxisCount,edgeNeighbouringCubes,edgeIntersectionNormals,cubeVertices)



vXY=interpolateList_vNormalXY(starMeshVertices,starMeshVertexNormals,particleTree,particleMass,particleH,particlePositions,particleVelocities)
vR=interpolateList_vNormalR(starMeshVertices,starMeshVertexNormals,particleTree,particleMass,particleH,particlePositions,particleVelocities)
print("maximum minimum XY "+str(max(vXY))+" "+str(min(vXY)))
print("maximum minimum R "+str(max(vR))+" "+str(min(vR)))


print("Creating UV map for star.")
starMeshUvR,starMeshUvG,starMeshUvB,starMeshUvZ0,starMeshFacesUv=determineEquirectangularUvPositions(numpy.array([0.0,0.0,0.0]),starMeshVertices,starMeshFaces,vXY,vXYmin,vXYmax)
print("Creating surface texture.")
fillTexture(starMeshUvR,starMeshUvG,starMeshUvB,starMeshUvZ0,starMeshFacesUv,textureHeight,outputFilename+"_vXY")

print("Creating UV map for star.")
starMeshUvR,starMeshUvG,starMeshUvB,starMeshUvZ0,starMeshFacesUv=determineEquirectangularUvPositions(numpy.array([0.0,0.0,0.0]),starMeshVertices,starMeshFaces,vR,vRmin,vRmax)
print("Creating surface texture.")
fillTexture(starMeshUvR,starMeshUvG,starMeshUvB,starMeshUvZ0,starMeshFacesUv,textureHeight,outputFilename+"_vR")



meshData=[["Star isosurface "+"("+outputFilename+")",starMeshVertices,starMeshUvZ0,starMeshVertexNormals,starMeshFaces,starMeshFacesUv]]

for i in range(0,sinkPositions.shape[0]):
    svp,svn,sf=createSphereMeshData(sinkPositions[i,:],10.0,6,10)
    meshData.append(["Sink "+str(i) +" ("+outputFilename+")",svp,None,svn,sf,None])

print("Writing to .obj file.")
writeObjFile(outputFilename,meshData)
