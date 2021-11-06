#This file implements parts of "Dual Contouring of Hermite Data" (https://www.cs.wustl.edu/~taoju/research/dualContour.pdf)
#and "Dual Marching Tetrahedra: Contouring in the Tetrahedronal Environment" (https://www.researchgate.net/publication/220845354_Dual_Marching_Tetrahedra_Contouring_in_the_Tetrahedronal_Environment).

import numpy
from sphInterpolation import dW,getNeighbouringParticleIndices


#Determines if the surface intersects the edge between the points v1 and v2 due to one side being below a threshold and the other over.
#Returns 0 if no intersection, -1 if it is an intersection passing the low threshold and 1 if it is an intersection passing the high threshold.
def intersectionOccursAtEdge(v1,v2,thresholdLow,thresholdHigh):
    lowThresholdIntersection=((v1<thresholdLow)and(v2>thresholdLow))or((v1>thresholdLow)and(v2<thresholdLow)) #An intersection occurs if one side is below the threshold and the other over.
    highThresholdIntersection=((v1<thresholdHigh)and(v2>thresholdHigh))or((v1>thresholdHigh)and(v2<thresholdHigh))
    
    #No intersection is detected when there is no intersection across either thresholds or when there is an intersection across both. When there
    #is an intersection across both could mean that there is structure below the resolution of the grid, and as it can not be resolved it is ignored.
    noIntersection=not(lowThresholdIntersection^highThresholdIntersection)

    if(noIntersection):
        return 0
    elif(lowThresholdIntersection):
        return -1
    else: #When highThresholdIntersection is true.
        return 1

#Determines how far along an edge the surface intersects using linear interpolation.
def determineEdgeIntersection(c1,c2,v1,v2,threshold):
    fractionAlongEdge=(threshold-v1)/(v2-v1) #The fraction that the intersection point lies between c1 and c2.
    intersectionPoint=c1+(fractionAlongEdge*(c2-c1))
    return intersectionPoint
        
   

#Determines the normal vector of the density field at a particular point.
def determineNormalAtPoint(point,flipNormals,particleTree,particleMass,particleH,particlePositions):    
    neighbouringParticleIndices=getNeighbouringParticleIndices(numpy.array([point]),particleTree,particleH)[0]
    neighbouringParticleH=particleH[neighbouringParticleIndices]
    neighbouringParticlePositions=particlePositions[neighbouringParticleIndices]

    dWvalues=dW(neighbouringParticleH,neighbouringParticlePositions,point) #An array of contributions to dW from all of the neighbouring SPH particles
    dWsum=numpy.sum(dWvalues,axis=0)
    gradient=dWsum*particleMass

    if(flipNormals):
        gradient*=-1.0 #Normals need to be flipped on low threshold intersections because the gradient points from low values to high values.
    
    return gradient/numpy.linalg.norm(gradient) #The gradient is normalized to give a normal vector.




#Determines is the surface intersects the edges of the tetrahedra inside the cubes of the sampling grid, and puts the interpolated intersection points their surface normals into lists.
def processEdgeIntersections(lowThreshold,highThreshold,
                             densities,samplingGridPositions,cubeAxisCount,
                             particleTree,particleMass,particleH,particlePositions): 
    
    #A list is created to hold the point along the edges of the grids where the surface intersects for the tetrahedtrons inside each grid cube. Indexed by grid point location and then 
    #by one of seven unique directions for the edge.
    edgeIntersections=numpy.zeros(shape=(cubeAxisCount,cubeAxisCount,cubeAxisCount,7,3))

    #A list is created to hold the normal of the surface where it intersects edges.
    edgeIntersectionNormals=numpy.zeros(shape=(cubeAxisCount,cubeAxisCount,cubeAxisCount,7,3))
    
    
    #Edge directions in order: x,y,z,xz diagonal, yz diagonal, xy diagonal, interior cube edge. This is a list of corner pairs for each of these edges relative to the grid
    #cube origin points.
    edgeCornerPairs=[[[0,0,0],[1,0,0]], #x 
                     [[0,0,0],[0,1,0]], #y
                     [[0,0,0],[0,0,1]], #z 
                     [[1,0,0],[0,0,1]], #xz diagonal 
                     [[0,1,0],[0,0,1]], #yz diagonal
                     [[0,0,0],[1,1,0]], #xy diagonal
                     [[0,0,1],[1,1,0]]] #interior cube edge
    
    
    #For each of the 7 edge types stores the relative tetrahedron positions that share that edge. The first three ordinates are the relative grid cube position that contains
    #the tetrahedron, and the last ordinate is the index representing the tetrahedron inside that grid cube. 
    edgeNeighbouringTetrahedraRelative=[[[0,0,0,1],[0,-1,0,3],[0,-1,0,2],[0,-1,-1,4],[0,0,-1,0],[0,0,-1,5]], #x 
                                        [[0,0,0,2],[-1,0,0,0],[-1,0,0,1],[-1,0,-1,5],[0,0,-1,3],[0,0,-1,4]], #y
                                        [[0,0,0,1],[0,0,0,2],[-1,0,0,0],[-1,-1,0,4],[-1,-1,0,5],[0,-1,0,3]], #z
                                        [[0,0,0,0],[0,0,0,1],[0,-1,0,3],[0,-1,0,4]], #xz diagonal 
                                        [[0,0,0,2],[0,0,0,3],[-1,0,0,5],[-1,0,0,0]], #yz diagonal 
                                        [[0,0,0,1],[0,0,0,2],[0,0,-1,4],[0,0,-1,5]], #xy diagonal
                                        [[0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,3],[0,0,0,4],[0,0,0,5]]] #interior cube edge. 
    
    #Holds the coordinates of each neighbouring tetrahedron that share each edge.
    edgeNeighbouringTetrahedra=[[[[[] for ed in range(0,7)] for z in range(0,cubeAxisCount)] for y in range(0,cubeAxisCount)] for x in range(0,cubeAxisCount)]
    
    #Holds the indices of edges that intersect the surface of each tetrahedron.
    tetrahedronEdgeIntersections=[[[[[]for ti in range(0,6)] for z in range(0,cubeAxisCount)] for y in range(0,cubeAxisCount)] for x in range(0,cubeAxisCount)]
        
    
    for x in range(0,cubeAxisCount): #Loops over all grid cube origin points.
        for y in range(0,cubeAxisCount):
            for z in range(0,cubeAxisCount):
                for ed,(c1,c2) in enumerate(edgeCornerPairs): #Loops through all seven edge corner pairs.
                    c1Position=samplingGridPositions[x+c1[0]][y+c1[1]][z+c1[2]]
                    c2Position=samplingGridPositions[x+c2[0]][y+c2[1]][z+c2[2]]
                    c1Density=densities[x+c1[0],y+c1[1],z+c1[2],:]
                    c2Density=densities[x+c2[0],y+c2[1],z+c2[2],:]
                    
                    currentEdgeIntersectionType=intersectionOccursAtEdge(c1Density,c2Density,lowThreshold,highThreshold)
                    if(currentEdgeIntersectionType!=0): #If the surface makes some sort of intersection along the current edge.
                        flipNormals=(currentEdgeIntersectionType==-1) #Done to make sure the face normals are the correct way around where low threshold intersections occur.
                        intersectionThresholdValue=lowThreshold if(currentEdgeIntersectionType==-1) else highThreshold
                    
                        edgeIntersections[x,y,z,ed]=determineEdgeIntersection(c1Position,c2Position,c1Density,c2Density,intersectionThresholdValue)
                        edgeIntersectionNormals[x,y,z,ed]=determineNormalAtPoint(edgeIntersections[x][y][z][ed],flipNormals,particleTree,particleMass,particleH,particlePositions)
                        
                        #Below the current edge is assigned its neighbouring tetrahedra and the neighbouring tetrahedra are assigned the current edge as having an intersection.
                        currentEdgeNeighbouringTetrahedraRelativeIndices=edgeNeighbouringTetrahedraRelative[ed]
                        for gcx_r,gcy_r,gcz_r,gct in currentEdgeNeighbouringTetrahedraRelativeIndices: #Loops over all tetrahedra sharing this edge.
                            gcx=x+gcx_r #Non relative grid cube indices for the neighbouring tetrahedra.
                            gcy=y+gcy_r
                            gcz=z+gcz_r
                            
                            tetrahedronEdgeIntersections[gcx][gcy][gcz][gct].append([x,y,z,ed]) #The current tetrahedron is assigned this edge as having an intersection.
                            edgeNeighbouringTetrahedra[x][y][z][ed].append([gcx,gcy,gcz,gct]) #The current edge is assigned the current tetrahedron as a neighbour.
                                                       
                            
    return edgeIntersections,edgeIntersectionNormals,edgeNeighbouringTetrahedra,tetrahedronEdgeIntersections
  
                
                



#Determines the optimal position for a vertex inside each tetrahedra with surface edge intersections. The normal of the vertex is
#also obtained.
def determineVertices(tetrahedraEdgeIntersections,samplingGridPositions,cubeAxisCount,cubeSize,
                      edgeIntersections,edgeIntersectionNormals,
                      particleTree,particleMass,particleH,particlePositions,interpolatedMesh):
    
    #Holds coordinates and normals of the mesh vertices for the grid cubes that have a vertex.
    tetrahedronVertices=[[[[None for t in range(0,6)] for z in range(0,cubeAxisCount)] for y in range(0,cubeAxisCount)] for x in range(0,cubeAxisCount)]
    
    #The relative centres of each of the 6 tetrahedrons inside each grid cube.
    tetrahedronCentres=[numpy.array([0.75,0.25,0.5])*cubeSize,
                        numpy.array([0.5,0.25,0.25])*cubeSize,
                        numpy.array([0.25,0.5,0.25])*cubeSize,
                        numpy.array([0.25,0.75,0.5])*cubeSize,
                        numpy.array([0.5,0.75,0.75])*cubeSize,
                        numpy.array([0.75,0.5,0.75])*cubeSize]
        
    for x in range(0,cubeAxisCount): #Loops over all grid cubes.
        for y in range(0,cubeAxisCount):
            for z in range(0,cubeAxisCount):
                for t in range(0,6): #Loops over each tetrahedron in a grid cube.
                    currentTetrahedronIntersections=tetrahedraEdgeIntersections[x][y][z][t]
                
                    if(len(currentTetrahedronIntersections)!=0): #The the current tetrahedron has edges that have been intersected, a vertex will be placed inside it.
                        intersectedEdges=numpy.array(currentTetrahedronIntersections) #An array of the indices of intersected edges for the tetrahedron.
                        intersectedEdgesX=intersectedEdges[:,0]
                        intersectedEdgesY=intersectedEdges[:,1]
                        intersectedEdgesZ=intersectedEdges[:,2]
                        intersectedEdgesE=intersectedEdges[:,3]
                    
                        #Arrays of the edge intersection locations and surface normals.
                        intersectedEdgePositions=edgeIntersections[intersectedEdgesX,intersectedEdgesY,intersectedEdgesZ,intersectedEdgesE,:]
                        intersectedEdgeNormals=edgeIntersectionNormals[intersectedEdgesX,intersectedEdgesY,intersectedEdgesZ,intersectedEdgesE,:]
    
                        #The vertex is initally placed in the centre of the tetrahedron.
                        currentVertexPosition=samplingGridPositions[x][y][z]+tetrahedronCentres[t]
                    
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
                        
                     
                        averageIntersectingEdgeNormals=numpy.average(intersectedEdgeNormals,axis=0) #This is the average direction of the intersecitng edge normals for this vertex.
                        currentVertexNormal=determineNormalAtPoint(currentVertexPosition,False,particleTree,particleMass,particleH,particlePositions)
                        if(numpy.dot(averageIntersectingEdgeNormals,currentVertexNormal)<0.0): #The vertex normal needs to be flipped if it is representing a surface bounding the low threshold value.
                            currentVertexNormal*=-1.0 #The normal is flipped to align with the general direction of the surrounding edge intersection normals if they do not already align.
                    
                        tetrahedronVertices[x][y][z][t]=interpolatedMesh.addVertex(currentVertexPosition,currentVertexNormal)
                    
    return tetrahedronVertices
                    
    
    
#For each tetrahedral edge that intersects the isosurface a face is created. This face consists of the vertices inside all of the tetrahedrons that
#share the intersected edge.
def assignMeshFaces(cubeAxisCount,edgeNeighbouringTetrahedra,tetrahedronVertices,interpolatedMesh):
    for x in range(0,cubeAxisCount): #Loops over all grid cubes.
        for y in range(0,cubeAxisCount):
            for z in range(0,cubeAxisCount):              
                for ed in range(0,7): #Loops over all edges for the orgin of each grid cube.
                    currentEdgeNeighbouringTetrahedra=edgeNeighbouringTetrahedra[x][y][z][ed]
                    
                    if(len(currentEdgeNeighbouringTetrahedra)!=0): #If the current intersected edge has surrounding vertices.
                        faceVertexList=[]
                        for gcx,gcy,gcz,gct in currentEdgeNeighbouringTetrahedra:
                            faceVertexList.append(tetrahedronVertices[gcx][gcy][gcz][gct])

                        interpolatedMesh.addFaceToWaitingList(faceVertexList)
                                                       
                                      