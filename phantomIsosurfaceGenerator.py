import glob
import os
import yaml
import h5py
import math
import numpy
from sklearn.neighbors import KDTree
from joblib import Parallel,delayed

from interpolatedValues import interpolateGrid_density,interpolateList_vNormalXY,interpolateList_vNormalR
from dualContouring import processEdgeIntersections,determineVertices,assignMeshFaces
from uvMapping import determineEquirectangularUvPositions,determineUvVertexColours,fillTexture
from halfedge import Mesh,writeMeshesToObjFile


#Creates a material file so the isosurface can have its texture loaded automatically in 3d modelling software.
#A material for the star's core and the planets are also created. In order to be used, the .mtl file needs to
#be renamed so it has the same name as the .obj file (with a .mtl file extension instead).
def writeMaterialFile(filename,isosurfaceTextureName):
    mtlFile=open(filename+".mtl","wt")
    
    #The isosurface material with its texture.
    mtlFile.write("newmtl IsosurfaceTexture\n")
    mtlFile.write("Ka 1.0 1.0 1.0\n") #Ambient colour.
    mtlFile.write("Kd 1.0 1.0 1.0\n") #Diffuse colour.
    mtlFile.write("illum 1\n") #Turn ambient and colour rendering on.
    mtlFile.write("map_Ka "+isosurfaceTextureName+".png\n") #Ambient texture map.
    mtlFile.write("map_Kd "+isosurfaceTextureName+".png\n") #Diffuse texture map.
    
    #The stellar core material (solid red colour).
    mtlFile.write("newmtl CoreColour\n")
    mtlFile.write("Ka 1.0 0.0 0.0\n")
    mtlFile.write("Kd 1.0 0.0 0.0\n")
    mtlFile.write("illum 1\n")
    
    #The planet material (solid blue colour).
    mtlFile.write("newmtl PlanetColour\n")
    mtlFile.write("Ka 0.0 0.0 1.0\n")
    mtlFile.write("Kd 0.0 0.0 1.0\n")
    mtlFile.write("illum 1\n")
    
    mtlFile.close()
   
    
#Creates the mesh data for a sphere.    
def createSphereMeshData(centre,radius,zDivisions,xyDivisions,meshName):
    sphereMesh=Mesh(meshName)
       
    #The vertex positions and normals are added.
    for phi in numpy.linspace(0.5*math.pi,(-0.5)*math.pi,zDivisions): #Vertical angle.
        if(phi==(0.5*math.pi)): #North pole.
            sphereMesh.addVertex(centre+numpy.array([0.0,0.0,radius]),numpy.array([0.0,0.0,1.0]))
            continue
        
        if(phi==((-0.5)*math.pi)): #South pole.
            sphereMesh.addVertex(centre+numpy.array([0.0,0.0,(-1.0)*radius]),numpy.array([0.0,0.0,-1.0]))
            continue
        
        z=centre[2]+(radius*math.sin(phi))
        currentVertexRingRadius=radius*math.cos(phi) #Radius of vertex ring at the current height from the sphere's centre.
               
        for theta in numpy.linspace(0.0,2.0*math.pi,xyDivisions,endpoint=False): #Horizontal angle.
            x=centre[0]+(currentVertexRingRadius*math.cos(theta))
            y=centre[1]+(currentVertexRingRadius*math.sin(theta))
            
            currentVertexPosition=numpy.array([x,y,z])
            sphereMesh.addVertex(currentVertexPosition,(currentVertexPosition-centre)/radius)
            
            
    #The faces are added.
    for phiIndex in range(1,zDivisions):
        for thetaIndex in range(0,xyDivisions):
            #The indices of vertices to make up the current face/s to be added.
            bottomRightIndex=2+thetaIndex+(xyDivisions*(phiIndex-1))
            bottomLeftIndex=bottomRightIndex+(xyDivisions-1) if(thetaIndex==0) else bottomRightIndex-1 #At the start of a vertex ring the vertex index immediently to the left was the last one added in that ring.
            topRightIndex=bottomRightIndex-xyDivisions #Non pole vertices immediately above or below a non pole vertex are xyDivisions positions away.  
            topLeftIndex=bottomLeftIndex-xyDivisions
            
            numberOfVertices=(xyDivisions*(zDivisions-2))+2
            bottomRightVertex=sphereMesh.vertices[min(bottomRightIndex-1,numberOfVertices-1)] #The min function stops the list index from going out of bounds while triangulating the south pole 
            bottomLeftVertex=sphereMesh.vertices[min(bottomLeftIndex-1,numberOfVertices-1)] #as the bottom right and bottom left vertices are not needed for it.
            topRightVertex=sphereMesh.vertices[topRightIndex-1]
            topLeftVertex=sphereMesh.vertices[topLeftIndex-1]
            
            
            if(phiIndex==1): #The first vertex ring below the north pole.
                northPoleVertex=sphereMesh.vertices[0]
                sphereMesh.addFaceToWaitingList([bottomLeftVertex,bottomRightVertex,northPoleVertex]) #A single triangle is added attached to the north pole.
            elif(phiIndex==(zDivisions-1)): #The south pole.
                southPoleVertex=sphereMesh.vertices[numberOfVertices-1]
                sphereMesh.addFaceToWaitingList([southPoleVertex,topRightVertex,topLeftVertex]) #A single triangle is added attached to the south pole.
            else: #When another vertex ring is above the current one it creates a strip of rectangular faces.
                sphereMesh.addFaceToWaitingList([bottomLeftVertex,bottomRightVertex,topRightVertex,topLeftVertex])
    
    sphereMesh.createMesh()
    return sphereMesh
            
                                   
            
#Creates a grid of coordinates where the densities and density gradients will be sampled.
def createSamplingGridPositions(cubeAxisCount,cubeSize): 
    cornerPosition=(float(cubeAxisCount)*cubeSize)/(2.0) #The grid is centred at the position 0,0,0.
    axesOrdinates=numpy.linspace((-1.0*cornerPosition),cornerPosition,cubeAxisCount+1)
    axesXarray,axesYarray,axesZarray=numpy.meshgrid(axesOrdinates,axesOrdinates,axesOrdinates,indexing="ij")
    outputGrid=numpy.stack((axesXarray,axesYarray,axesZarray),axis=3) #This is a 3d array of vectors that describe the location of each point in the 3d grid.              
    return outputGrid







#Processes a single .h5 input file in order to create a isosurface model and textures.
def oneIsosurfaceGeneration(configurationData):
    inputFilename=configurationData["inputFilename"]
    inputFileFolder=os.getcwd()+configurationData["inputFileFolder"]+"\\"
    
    outputFilenamePrefix=configurationData["outputFilenamePrefix"]
    outputFileFolder=os.getcwd()+configurationData["outputFileFolder"]+"\\"
    
    cubeAxisCount=configurationData["cubeAxisCount"] #Number of cubes to split the area up in all three axes. Number of sample points in an axis is equal to this number plus 1.
    cubeSize=configurationData["cubeSize"] #Width of cube side length
    
    textureHeight=configurationData["textureHeight"] #Does not correspond exactly to height in pixels.
    
    isosurfaceDensityLowThreshold=configurationData["isosurfaceDensityLowThreshold"] #This low value forms an isosurface boundary.
    isosurfaceDensityHighThreshold=configurationData["isosurfaceDensityHighThreshold"] #This high value forms an isosurface boundary.
    
    #Below are the minimum and maximum velocities the textures will represent.
    vXYmin=configurationData["vXYmin"]
    vXYmax=configurationData["vXYmax"]
    vRmin=configurationData["vRmin"]
    vRmax=configurationData["vRmax"]
    
    
    def printWithInputName(stringToPrint): #This is done so you can see what input file the message corresponds to when processing multiple input files at once with joblib.
        print("("+inputFilename+") "+stringToPrint)
        


 
    loadedFile=h5py.File(inputFileFolder+inputFilename+".h5","r")
    particleMass=loadedFile["header"]["massoftype"][0] #The mass of each SPH particle.
    particleHFactor=loadedFile["header"]["hfact"][()] #The smoothing length factor for the SPH particles.
    particleVelocities=numpy.array(loadedFile["particles"]["vxyz"])
    sinkPositions=loadedFile["sinks"]["xyz"]

    particleH=numpy.array(loadedFile["particles"]["h"])
    particlePositions=numpy.array(loadedFile["particles"]["xyz"])
    particleTree=KDTree(particlePositions) #Stores particle positions in a kd tree so the neighbouring particles to a sampling location can be quickly found.
    outputMeshes=[] #A list of meshes to write to a single .obj file.


    printWithInputName("Creating sampling grid.")
    samplingGridPositions=createSamplingGridPositions(cubeAxisCount,cubeSize)
    printWithInputName("Interpolating density grid.")
    densities=interpolateGrid_density(samplingGridPositions,particleTree,particleMass,particleH,particlePositions)

    printWithInputName("Finding grid-isosurface intersections.")
    edgeIntersections,edgeIntersectionNormals,edgeNeighbouringTetrahedra,tetrahedronEdgeIntersections=processEdgeIntersections(isosurfaceDensityLowThreshold,isosurfaceDensityHighThreshold,
                                                                                                                               densities,samplingGridPositions,cubeAxisCount,
                                                                                                                               particleTree,particleMass,particleH,particlePositions)
    
    printWithInputName("Placing vertices.")
    interpolatedMesh=Mesh("Isosurface ("+outputFilenamePrefix+inputFilename+")")
    tetrahedronVertices=determineVertices(tetrahedronEdgeIntersections,samplingGridPositions,cubeAxisCount,cubeSize,
                                          edgeIntersections,edgeIntersectionNormals,
                                          particleTree,particleMass,particleH,particlePositions,interpolatedMesh)

    printWithInputName("Creating faces of isosurface mesh.")
    assignMeshFaces(cubeAxisCount,edgeNeighbouringTetrahedra,tetrahedronVertices,interpolatedMesh)
    interpolatedMesh.createMesh()

    printWithInputName("Triangulating faces")
    interpolatedMesh.triangulateAllFaces()
    interpolatedMesh.mtlFilename,interpolatedMesh.materialName=outputFilenamePrefix+inputFilename+".mtl","IsosurfaceTexture" #The material for the isosurface model is set.
    outputMeshes.append(interpolatedMesh)


    printWithInputName("Determining velocities at mesh surface.")
    printWithInputName("Interpolating normal-XY velocity.")
    vXY=interpolateList_vNormalXY([cv.position for cv in interpolatedMesh.vertices],[cv.normal for cv in interpolatedMesh.vertices],particleTree,particleMass,particleH,particlePositions,particleVelocities,particleHFactor)
    printWithInputName("Interpolating normal-R velocity.")
    vR=interpolateList_vNormalR([cv.position for cv in interpolatedMesh.vertices],[cv.normal for cv in interpolatedMesh.vertices],particleTree,particleMass,particleH,particlePositions,particleVelocities,particleHFactor)
    printWithInputName("  maximum minimum XY "+str(max(vXY))+" "+str(min(vXY)))
    printWithInputName("  maximum minimum R "+str(max(vR))+" "+str(min(vR)))

    printWithInputName("Creating equirectangular projected UV map.")
    determineEquirectangularUvPositions(numpy.array([0.0,0.0,0.0]),interpolatedMesh)
    printWithInputName("Creating XY surface texture.")
    XYtextureFilename=outputFilenamePrefix+inputFilename+"_vXY"
    determineUvVertexColours(interpolatedMesh,vXY,vXYmin,vXYmax)
    fillTexture(interpolatedMesh,textureHeight,outputFileFolder+XYtextureFilename)
    writeMaterialFile(outputFileFolder+"vXY_"+outputFilenamePrefix+inputFilename,XYtextureFilename)
    
    printWithInputName("Creating R surface texture.")
    RtextureFilename=outputFilenamePrefix+inputFilename+"_vR"
    determineUvVertexColours(interpolatedMesh,vR,vRmin,vRmax)
    fillTexture(interpolatedMesh,textureHeight,outputFileFolder+RtextureFilename)
    writeMaterialFile(outputFileFolder+"vR_"+outputFilenamePrefix+inputFilename,RtextureFilename)


    printWithInputName("Creating sink particle meshes.")
    for i in range(0,sinkPositions.shape[0]):
        currentMesh=createSphereMeshData(sinkPositions[i,:],10.0,6,10,"Sink "+str(i))      
        currentMesh.mtlFilename=outputFilenamePrefix+inputFilename+".mtl"
        currentMesh.materialName="CoreColour" if (i==0) else "PlanetColour" #The first sink is the star's core, while the others are planets.
        outputMeshes.append(currentMesh)

    printWithInputName("Writing to .obj file.")
    writeMeshesToObjFile(outputFileFolder+outputFilenamePrefix+inputFilename,outputMeshes)
    
    
    
def main():
    configurationFile=open("configuration.yaml",mode="r")
    configurationData=list(yaml.safe_load_all(configurationFile))[0]
    
    #Gets all the .h5 file filenames in the input file folder that match the globbing string in the configuration file.
    originalWorkingFolder=os.getcwd()
    os.chdir(originalWorkingFolder+configurationData["inputFileFolder"]) #The working directory is changed to the input file folder that is specified in the configuration file.
    inputGlobbingString=configurationData["inputFilename"]+".h5"
    inputFilenames=glob.glob(inputGlobbingString)
    os.chdir(originalWorkingFolder) #The current working directory is reset.
    
    #Creates the output folder if it doesn't already exist.
    outputFileFolder=os.getcwd()+configurationData["outputFileFolder"]
    if(os.path.isdir(outputFileFolder)==False):
        os.mkdir(outputFileFolder)
            
    #Below a copy of configurationData is created for each of the input filenames.
    individualConfigurationData=[dict(configurationData) for i in enumerate(inputFilenames)]
    for currentConfigurationData,currentFilename in zip(individualConfigurationData,inputFilenames):
        currentConfigurationData["inputFilename"]=currentFilename[0:-3] #The .h5 extension is removed on the right-hand side.
    
    #Multiple input files are processed in parallel if desired.
    Parallel(n_jobs=configurationData["ncpus"])(delayed(oneIsosurfaceGeneration)(ccd) for ccd in individualConfigurationData)
    print(str(len(individualConfigurationData))+" input files have been processed.")
        
   

    
if(__name__=="__main__"):
    main()
