import numpy
from sphInterpolation import dW


#Determines how far along an edge the surface intersects using linear interpolation.
def determineEdgeIntersection(c1,c2,v1,v2,threshold,axis):
    intersectionPoint=numpy.array(c1)
    
    if(axis=="x"):
        m=(v2-v1)/(c2[0]-c1[0])
        axisShift=((threshold-v1)/m)+c1[0]
        intersectionPoint[0]=axisShift
    elif (axis=="y"):
        m=(v2-v1)/(c2[1]-c1[1])
        axisShift=((threshold-v1)/m)+c1[1]
        intersectionPoint[1]=axisShift
    else:
        m=(v2-v1)/(c2[2]-c1[2])
        axisShift=((threshold-v1)/m)+c1[2]
        intersectionPoint[2]=axisShift

    return intersectionPoint
        
   

#Determines the normal vector of the density field at a particular point.
def determineNormalAtPoint(point,flipNormals,particleTree,particleMass,particleH,particlePositions):
    intersectionPointH=particleH[particleTree.query([point],1,return_distance=False).squeeze()] #The smoothing length of the closest SPH particle.
    neighbouringParticleIndices=particleTree.query_radius([point],2.0*intersectionPointH)[0] #The indices of all particles within 2 times the chosen smoothing length.
    neighbouringParticleH=particleH[neighbouringParticleIndices]
    neighbouringParticlePositions=particlePositions[neighbouringParticleIndices]
    
    dWvalues=dW(neighbouringParticleH,neighbouringParticlePositions,point) #An array of contributions to dW from all of the neighbouring SPH particles
    dWsum=numpy.sum(dWvalues,axis=0)
    gradient=dWsum*particleMass
    if(flipNormals):
        gradient*=-1.0 #Normal direction can be flipped depending on whether the interior of the surface is higher or lower than the threshold value.
    
    return gradient/numpy.linalg.norm(gradient) #The gradient is normalized to give a normal vector.
    

#Determines is the surface intersects the edges of cubes in the sampling grid, and puts the interpolated intersection points their surface normals into lists.    
def processEdgeIntersections(threshold,flipNormals,densities,samplingGridPositions,cubeAxisCount,
                             particleTree,particleMass,particleH,particlePositions):    
    
    #A list is created to hold the point along the edges of the grids where the surface intersects. Indexed by grid point location and then by one of three directions for the edge.
    edgeIntersections=numpy.zeros(shape=(cubeAxisCount,cubeAxisCount,cubeAxisCount,3,3))

    #A list is created to hold the normal of the surface where it intersects edges.
    edgeIntersectionNormals=numpy.zeros(shape=(cubeAxisCount,cubeAxisCount,cubeAxisCount,3,3))

    #Holds the indexes of cubes that share an edge.
    edgeNeighbouringCubes=[[[[None for ed in range(0,3)] for z in range(0,cubeAxisCount)] for y in range(0,cubeAxisCount)] for x in range(0,cubeAxisCount)]

    #Holds indexes of edges that intersect the surface for each cube.
    cubeEdgeIntersections=[[[[] for z in range(0,cubeAxisCount)] for y in range(0,cubeAxisCount)] for x in range(0,cubeAxisCount)]

    #Loops through one of the corners of each cube. Each of these corners has three perpendicular edges associated with it.
    for x in range(0,cubeAxisCount):
        for y in range(0,cubeAxisCount):
            for z in range(0,cubeAxisCount):
                #The position and density of the corner common to all three edges to be dealt with.
                cornerCommonPosition=samplingGridPositions[x][y][z]
                cornerCommonDensity=densities[x,y,z,:]
                
                cornerXPosition=samplingGridPositions[x+1][y][z]
                cornerXDensity=densities[x+1,y,z,:]
                xAxisIntersection=((cornerCommonDensity>threshold)and(cornerXDensity<threshold))or((cornerCommonDensity<threshold)and(cornerXDensity>threshold))
                #The surface intersects an edge if one corner is below the surface threshold and the other corner is above it.
                
                cornerYPosition=samplingGridPositions[x][y+1][z]
                cornerYDensity=densities[x,y+1,z,:]
                yAxisIntersection=((cornerCommonDensity>threshold)and(cornerYDensity<threshold))or((cornerCommonDensity<threshold)and(cornerYDensity>threshold))
                
                cornerZPosition=samplingGridPositions[x][y][z+1]
                cornerZDensity=densities[x,y,z+1,:]
                zAxisIntersection=((cornerCommonDensity>threshold)and(cornerZDensity<threshold))or((cornerCommonDensity<threshold)and(cornerZDensity>threshold))

                #Intersections with the three edges attatched to the common corner are dealt with if necessary.
                if(xAxisIntersection):
                    edgeIntersections[x,y,z,0]=determineEdgeIntersection(cornerCommonPosition,cornerXPosition,cornerCommonDensity,cornerXDensity,threshold,"x")
                    edgeIntersectionNormals[x,y,z,0]=determineNormalAtPoint(edgeIntersections[x][y][z][0],flipNormals,particleTree,particleMass,particleH,particlePositions)
                    
                    #All the cubes that share this edge will be set to contain a vertex.
                    cubeEdgeIntersections[x][y][z].append([x,y,z,0])
                    cubeEdgeIntersections[x][y][max(0,z-1)].append([x,y,z,0])
                    cubeEdgeIntersections[x][max(0,y-1)][z].append([x,y,z,0])
                    cubeEdgeIntersections[x][max(0,y-1)][max(0,z-1)].append([x,y,z,0])
                    
                    #The neighboring cubes of this edge are set.
                    edgeNeighbouringCubes[x][y][z][0]=[[x,y,z],[x,y,max(0,z-1)],[x,max(0,y-1),z],[x,max(0,y-1),max(0,z-1)]]
                    
                if(yAxisIntersection):
                    edgeIntersections[x,y,z,1]=determineEdgeIntersection(cornerCommonPosition,cornerYPosition,cornerCommonDensity,cornerYDensity,threshold,"y")
                    edgeIntersectionNormals[x,y,z,1]=determineNormalAtPoint(edgeIntersections[x][y][z][1],flipNormals,particleTree,particleMass,particleH,particlePositions)
                    
                    cubeEdgeIntersections[x][y][z].append([x,y,z,1])
                    cubeEdgeIntersections[x][y][max(0,z-1)].append([x,y,z,1])
                    cubeEdgeIntersections[max(0,x-1)][y][z].append([x,y,z,1])
                    cubeEdgeIntersections[max(0,x-1)][y][max(0,z-1)].append([x,y,z,1])
                    
                    edgeNeighbouringCubes[x][y][z][1]=[[x,y,z],[x,y,max(0,z-1)],[max(0,x-1),y,z],[max(0,x-1),y,max(0,z-1)]]
                    
                if(zAxisIntersection):
                    edgeIntersections[x,y,z,2]=determineEdgeIntersection(cornerCommonPosition,cornerZPosition,cornerCommonDensity,cornerZDensity,threshold,"z")
                    edgeIntersectionNormals[x,y,z,2]=determineNormalAtPoint(edgeIntersections[x][y][z][2],flipNormals,particleTree,particleMass,particleH,particlePositions)
                    
                    cubeEdgeIntersections[x][y][z].append([x,y,z,2])
                    cubeEdgeIntersections[x][max(0,y-1)][z].append([x,y,z,2])
                    cubeEdgeIntersections[max(0,x-1)][y][z].append([x,y,z,2])
                    cubeEdgeIntersections[max(0,x-1)][max(0,y-1)][z].append([x,y,z,2])
                    
                    edgeNeighbouringCubes[x][y][z][2]=[[x,y,z],[x,max(0,y-1),z],[max(0,x-1),y,z],[max(0,x-1),max(0,y-1),z]]
                    
    return edgeIntersections,edgeIntersectionNormals,edgeNeighbouringCubes,cubeEdgeIntersections
                


#Determines the optimal position for a vertex inside each grid cube with surface edge intersections. The normal of the vertex is
#also obtained.
def determineVertices(flipNormals,cubeEdgeIntersections,samplingGridPositions,cubeAxisCount,cubeSize,
                      edgeIntersections,edgeIntersectionNormals,
                      particleTree,particleMass,particleH,particlePositions):
    
    #Holds coordinates and normals of the mesh vertices for the grid cubes that have a vertex.
    cubeVertices=[[[None for z in range(0,cubeAxisCount)] for y in range(0,cubeAxisCount)] for x in range(0,cubeAxisCount)]

    vertexIndex=1
    #Loops over all grid cubes.
    for x in range(0,cubeAxisCount):
        for y in range(0,cubeAxisCount):
            for z in range(0,cubeAxisCount):
                currentCubeIntersections=cubeEdgeIntersections[x][y][z]
                
                if(len(currentCubeIntersections)!=0): #The the current grid cube has edges that have been intersected, a vertex will be placed inside it.
                    #print("dvp "+str(x)+" "+str(y)+" "+str(z))
                    intersectedEdges=numpy.array(currentCubeIntersections) #An array of the indices of intersected edges for the grid cube.
                    intersectedEdgesX=intersectedEdges[:,0]
                    intersectedEdgesY=intersectedEdges[:,1]
                    intersectedEdgesZ=intersectedEdges[:,2]
                    intersectedEdgesD=intersectedEdges[:,3]
                    
                    #Arrays of the edge intersection locations and surface normals.
                    intersectedEdgePositions=edgeIntersections[intersectedEdgesX,intersectedEdgesY,intersectedEdgesZ,intersectedEdgesD,:]
                    intersectedEdgeNormals=edgeIntersectionNormals[intersectedEdgesX,intersectedEdgesY,intersectedEdgesZ,intersectedEdgesD,:]
    
                    #The vertex is initally placed in the centre of the grid cube.
                    currentVertexPosition=samplingGridPositions[x][y][z]+numpy.array([cubeSize/2.0,cubeSize/2.0,cubeSize/2.0])
                    
                    #As the isosurface appears to be flat close up, each edge intersection can be considered to represent a infinite plane with a normal equal to the
                    #surface normal at the edge intersection point. Vertices should exist somewhere on the planes, and the single vertex inside the grid cube
                    #should be placed where all the planes intersect. Below, gradient descent is used to determine the optimal location for the vertex by
                    #minimising the sum of the squared perpendicular distances from the vertex to the surface planes. The formula for squared perpendiucular
                    #distance is (n.(x-p))^2, where n is the surface plane normal, p is the edge intersection point and x is the vertex position. The gradient of
                    #this formula is 2n(n.(x-p)) . The total gradient to be minimized is the sum of each gradients associated with an edge interection.
                    for i in range(0,10):
                        x_p=currentVertexPosition-intersectedEdgePositions #The current vertex poistion minus the edge intersection point for all edge intersections.
                        nXx_p=numpy.multiply(intersectedEdgeNormals,x_p) #Element wise multiplication.
                        nDotx_p=numpy.sum(nXx_p,axis=1) #The dot products between x_p and the normals for each edge intersection plane.
                        individualGradients=2.0*intersectedEdgeNormals*numpy.expand_dims(nDotx_p,axis=1) #The gradients for each squared perpendicular distance equation.
                        totalGradient=numpy.sum(individualGradients,axis=0) #The sum of all the individual gradients is the gradient of the function to be minimized.

                        currentVertexPosition-=(1.0/30.0)*totalGradient
                        
                        
                    currentVertexNormal=determineNormalAtPoint(currentVertexPosition,flipNormals,particleTree,particleMass,particleH,particlePositions)
                    cubeVertices[x][y][z]=(vertexIndex,currentVertexPosition,currentVertexNormal)
                    vertexIndex+=1
                    
    return cubeVertices
                    
    
    

#Returns lists of the vertex positions, vertex normals and faces that will make up the mesh of the surface. Faces are comprised
#of quadrilaterals split in two triangles made from the four vertices that surround an edge intersected by the surface.
def createMeshDataLists(cubeAxisCount,edgeNeighbouringCubes,edgeIntersectionNormals,cubeVertices):
    vertexPositions=[]
    vertexNormals=[]
    faces=[] #Contains the indices of the three vertices that make up the triangle.
    
    #Loops over all grid cubes.
    for x in range(0,cubeAxisCount):
        for y in range(0,cubeAxisCount):
            for z in range(0,cubeAxisCount):
                currentCubeVertex=cubeVertices[x][y][z]
                
                if(currentCubeVertex is not None):                
                    vertexPositions.append(currentCubeVertex[1])
                    vertexNormals.append(currentCubeVertex[2])
                
                for ed in range(0,3): #Loops over all edges for the orgin of each grid cube.
                    currentEdgeNeighbouringCubes=edgeNeighbouringCubes[x][y][z][ed]
                    
                    if(currentEdgeNeighbouringCubes is not None): #If the current intersected edge has surrounding vertices.
                        #The grid cube position indexes of the vertices to be turned into two triangles.
                        currentNeighbouringCubeIndex1=currentEdgeNeighbouringCubes[0]
                        currentNeighbouringCubeIndex2=currentEdgeNeighbouringCubes[1]
                        currentNeighbouringCubeIndex3=currentEdgeNeighbouringCubes[2]
                        currentNeighbouringCubeIndex4=currentEdgeNeighbouringCubes[3]
                        
                        vertex1=cubeVertices[currentNeighbouringCubeIndex1[0]][currentNeighbouringCubeIndex1[1]][currentNeighbouringCubeIndex1[2]]
                        vertex2=cubeVertices[currentNeighbouringCubeIndex2[0]][currentNeighbouringCubeIndex2[1]][currentNeighbouringCubeIndex2[2]]
                        vertex3=cubeVertices[currentNeighbouringCubeIndex3[0]][currentNeighbouringCubeIndex3[1]][currentNeighbouringCubeIndex3[2]]
                        vertex4=cubeVertices[currentNeighbouringCubeIndex4[0]][currentNeighbouringCubeIndex4[1]][currentNeighbouringCubeIndex4[2]]
                        
                        #The distances of the diagonals of the quadrilateral formed by the four neighbouring vertices. The quadrilateral will be
                        #split into two triangles across the shortest diagonal distance.
                        v1v4Distance=numpy.linalg.norm(vertex1[1]-vertex4[1])
                        v2v3Distance=numpy.linalg.norm(vertex2[1]-vertex3[1])
                        
                        if(v1v4Distance<v2v3Distance):
                            #From the outside, the vertex order for a face needs to be anti -lockwise for the face's normal to point outwards. By performing a cross
                            #product on vertices 4-1 and 1-2 the orientation of the vertex sequence 1,4,2 can be obtained, and after seeing if the dot product of the 
                            #result and the surface normal is greater than zero, it can be determined if the vertex order needs to be changed to the opposite direction.
                            vertexOrderNormalDirection=numpy.dot(numpy.cross(vertex4[1]-vertex1[1],vertex2[1]-vertex1[1]),edgeIntersectionNormals[x,y,z,ed])>0.0
                            if(vertexOrderNormalDirection):
                                faces.append([vertex1[0],vertex4[0],vertex2[0]])
                                faces.append([vertex4[0],vertex1[0],vertex3[0]])
                            else:
                                faces.append([vertex4[0],vertex1[0],vertex2[0]])
                                faces.append([vertex1[0],vertex4[0],vertex3[0]])
                        else:
                            vertexOrderNormalDirection=numpy.dot(numpy.cross(vertex3[1]-vertex2[1],vertex4[1]-vertex2[1]),edgeIntersectionNormals[x,y,z,ed])>0.0
                            if(vertexOrderNormalDirection):
                                faces.append([vertex2[0],vertex3[0],vertex4[0]])
                                faces.append([vertex3[0],vertex2[0],vertex1[0]])
                            else:
                                faces.append([vertex3[0],vertex2[0],vertex4[0]])
                                faces.append([vertex2[0],vertex3[0],vertex1[0]])
                                                        
    return vertexPositions,vertexNormals,faces
                    
                    