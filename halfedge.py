#This file implements the halfedge representation for 3d meshes.
#Uses information from https://martindevans.me/game-development/2016/03/30/Procedural-Generation-For-Dummies-Half-Edge-Geometry/

import math
import numpy


#Determies if a point is inside a triangle.
def pointInsideTriangle(v1,v2,v3,point):
    v1x,v1y=v1
    v2x,v2y=v2
    v3x,v3y=v3
    
    
    #Non-normalized normal vectors for each edge of the triangle. They are all either pointing outwards of the triangle or inwards; this depends on the winding order of v1,v2 and v3.
    n12=(v2y-v1y,-(v2x-v1x))
    n23=(v3y-v2y,-(v3x-v2x))
    n31=(v1y-v3y,-(v1x-v3x))
    
    #Below it is determined if the point lies inside (on the side opposite to the normal direction) each of the three edges using a dot product. 
    #The edges extend infinetely in both directions.
    insideOfE12=(n12[0]*(point[0]-v1x))+(n12[1]*(point[1]-v1y))<0.0
    insideOfE23=(n23[0]*(point[0]-v2x))+(n23[1]*(point[1]-v2y))<0.0
    insideOfE31=(n31[0]*(point[0]-v3x))+(n31[1]*(point[1]-v3y))<0.0
    
    #If the edge normals of a triangle all point outward, then a point is inside the triangle if the point is inside all three edges. If the edge normals all point inward then
    #the point is inside the triangle if the point is outside all of the edges.
    pointIsInside1=insideOfE12 and insideOfE23 and insideOfE31 #When the edge normals point outwards.
    pointIsInside2=(not insideOfE12) and (not insideOfE23) and (not insideOfE31) #When the edge normals point inwards.
    return pointIsInside1 or pointIsInside2


#Determines the signed area of a polygon depening on the order of vertices as viewed from the positive z axis. If the polygon is oriented clockwise then 
#the consecutive right hand going pairs of vertices will be higher than the consecutive left hand going pairs of vertices. The trapezoidal areas under each 
#consecutive pair or right hand going vertices are positive, and the trapezoidal areas under each consecutive pair of left hand going vertices are 
#negative (due to x2-x1 having an opposite sign). By adding these positive and negative areas together, the positive area components that are below the 
#polygon's bounds are removed by the addition of the negative areas. If the polygon is oriented anticlockwise then the left hand going pairs of vertices 
#are above the right hand going pairs of vertices, meaning that the area's magnitude is the same but negative. 
def getAreaOfPolygon(vertices):
    vertices2=list(numpy.roll(vertices,-1,axis=0)) #The list of vertices that are next relative to the elements in the original list.

    area=0
    for (x1,y1),(x2,y2) in zip(vertices,vertices2):
        currentTrapeziumArea=(x2-x1)*(y1+y2)*0.5
        area+=currentTrapeziumArea
    return area
        
    

class Vertex:
    def __init__(self,position,normal,listIndex):
        self.position=position #The vertices' position vector.
        self.normal=normal #The vertices normal vector.
        self.listIndex=listIndex #The position of this vertex in the mesh vertex list.
        self.halfedges=[] #A list of halfedges starting at this vertex.
        self.uvVertices={} #Holds UV vertices associated with this vertex with faces as keys.
    
    #Returns the halfedge starting at this vertex and ending at otherVertex if it exists.
    def getHalfedgeTo(self,otherVertex):
        for currentHalfedge in self.halfedges:
            if(currentHalfedge.endVertex is otherVertex):
                return currentHalfedge          
        return None
    
    #Finds the faces that use this vertex.
    def getFaces(self):
        faces=[]
        attachedHalfedges=self.halfedges+[currentHalfedge.oppositeHalfedge for currentHalfedge in self.halfedges] #Includes the halfedges that both begin and and at this vertex.
        
        for currentHalfedge in attachedHalfedges:
            currentHalfedgeFace=currentHalfedge.face
            
            if(currentHalfedgeFace):
                faces.append(currentHalfedgeFace)
        return faces

    
    


#Holds data for a UV map vertex.   
class UvVertex:
    def __init__(self,u,v,listIndex):
        self.position=numpy.array([u,v,0.0])
        self.listIndex=listIndex
        self.r=None #The colours are stored as 3d numpy arrays with the UV position as the first two components to make colour interpolation across triangles easier.
        self.g=None
        self.b=None        
        
    def setColour(self,r,g,b):
        self.r=numpy.array([self.position[0],self.position[1],r])
        self.g=numpy.array([self.position[0],self.position[1],g])
        self.b=numpy.array([self.position[0],self.position[1],b])
        
        
    
    
class Halfedge:
    def __init__(self,endVertex):
        self.endVertex=endVertex #The vertex that this half edge ends on.
        self.oppositeHalfedge=None #The halfedge that occupies the same space but goes in the opposite direction.
        self.face=None #The face attached to this halfedge.
        self.nextFaceHalfedge=None #The next halfedge that this halfedge is attached to around this halfedge's face.
        
    #Returns all halfedges from this one that are joined in a face loop.    
    def getHalfedgeFaceLoop(self):
        halfedges=[self]
        currentHalfedge=self.nextFaceHalfedge
        
        #Loops around all joined halfedges until it gets back to itself.
        while(currentHalfedge is not self):
            halfedges.append(currentHalfedge)
            currentHalfedge=currentHalfedge.nextFaceHalfedge
 
        return halfedges
        
    #The angle between this halfadge's face and the face of the opposite halfedge (if they exist).
    def angleBetweenFaces(self):
        if(self.face and self.oppositeHalfedge.face):
            return math.acos(numpy.dot(self.face.normal,self.oppositeHalfedge.face.normal)) #From the definition of the dot product.
        else:
            return None
        
        
        
class Face:
    def __init__(self,halfedge):
        self.halfedge=halfedge #A half edge that this face is attached to.
        self.normal=None #The normal of the face is the normalized average of the vertex normals.
        self.traversalFlag=False #A flag that turns true while recursively exploring connected faces.
    
    #Gets the half edges that bound the face.
    def getAllHalfedges(self):
        return self.halfedge.getHalfedgeFaceLoop()
    
    
    def getAllVertices(self):
        halfedges=self.getAllHalfedges()
        return [currentHalfedge.endVertex for currentHalfedge in halfedges]
    
    def getAllUvVertices(self):
        vertices=self.getAllVertices()
        return [currentVertex.uvVertices[self] for currentVertex in vertices]
    
    def getConnectedFaces(self):
        halfedges=self.getAllHalfedges()
        return [currentHalfedge.oppositeHalfedge.face for currentHalfedge in halfedges]   
        
    #Sets the normal of the face.    
    def calculateNormal(self):
        vertices1=self.getAllVertices()
        vertices2=list(numpy.roll(vertices1,-1))
        vertices3=list(numpy.roll(vertices1,-2))
        
        #Below the normal is first determined from averaging the cross product of every pair of edges.
        currentPlaneNormal=numpy.array([0.0,0.0,0.0])
        for v1,v2,v3 in zip(vertices1,vertices2,vertices3):
            currentCrossProduct=numpy.cross(v3.position-v2.position,v2.position-v1.position)
            currentCrossProduct/=numpy.linalg.norm(currentCrossProduct)
            currentPlaneNormal+=currentCrossProduct
        currentPlaneNormal/=numpy.linalg.norm(currentPlaneNormal)
            
        #A normal is then determined from the vertex normals. This is used to correct the orientation of the above type of normal. This type of normal is
        #not used by itself as it may give a poor estimation of the face's normal if the vertex normals are not aligned with the planes defined by the edge pairs.
        halfedges=self.getAllHalfedges()
        vertexNormals=[currentHalfedge.endVertex.normal for currentHalfedge in halfedges]
        
        averageVertexNormal=numpy.average(vertexNormals,axis=0)
        averageVertexNormal/=numpy.linalg.norm(averageVertexNormal)
        
        if(numpy.dot(currentPlaneNormal,averageVertexNormal)<0.0): #The edge pair based normal is flipped to align with the vertex normal if necessary.
            currentPlaneNormal*=-1.0
        self.normal=currentPlaneNormal
        

        
    #Returns the vertex positions projected onto a plane defined by the face's normal vector and an origin at the position 0,0,0.
    def getNormalPlaneProjection(self):
        #Two perpendicular vectors representing the tangent plane are obtained.
        startVector=numpy.array([0.707,0.707,0.0]) if(not numpy.array_equal(self.normal,numpy.array([0.707,0.707,0.0]))) else numpy.array([0.0,0.707,0.707])
        aLocal=numpy.cross(self.normal,startVector) #Perpendicular to the normal vector.     
        bLocal=numpy.cross(self.normal,aLocal) #Perpendicular to both the normal vector and aLocal.
        aLocal/=numpy.linalg.norm(aLocal)
        bLocal/=numpy.linalg.norm(bLocal)

        projectedCoordinates=[]
        for currentVertex in self.getAllVertices():
            normalProjection=self.normal*numpy.dot(self.normal,currentVertex.position) #The vertex position vector projected onto the tangent plane's normal vector.
            planeProjection=currentVertex.position-normalProjection #The projection of the vertex position onto the plane.
            
            #planeProjection is comprised of a linear combination of aLocal and bLocal. Below the coefficients are determined by solving two simultaneous equations.
            ppX,ppY=planeProjection[:2]
            aLocalX,aLocalY=aLocal[:2]
            bLocalX,bLocalY=bLocal[:2]

            y=(ppY-(ppX*(aLocalY/aLocalX)))/(bLocalY-(bLocalX*(aLocalY/aLocalX)))
            x=(ppX-(y*bLocalX))/aLocalX         
            projectedCoordinates.append((x,y))
            
        return projectedCoordinates
            
        
    #Returns whether the vertex winding order is anti-clockwise as viewed from the direction where the face's normal is pointing towards
    #the viewer If it is not then the vertex order needs to be swapped when writing to a .obj file.
    def windingOrderAndNormalAligned(self):
        v1=self.halfedge.nextFaceHalfedge.endVertex
        v2=self.halfedge.nextFaceHalfedge.nextFaceHalfedge.endVertex
        v3=self.halfedge.nextFaceHalfedge.nextFaceHalfedge.nextFaceHalfedge.endVertex
        
        windingDirection=numpy.cross(v2.position-v1.position,v3.position-v2.position) #The winding order appears anti-clockwise when viewed from above this direction.  
        return numpy.dot(self.normal,windingDirection)>0.0 #Is greater than zero if the normal of the first vertex aligns with the winding order.
        
        
        
        
        
class Mesh:  
    def __init__(self,name):
        self.name=name
        self.vertices=[]
        self.uvVertices=[]
        self.halfedges=[]
        self.faces=[]
        self.facesToCreate=[] #A list of vertex lists that represent faces to turned into a mesh.
        self.mtlFilename="" #The name of the material (.mtl) file to use for this mesh.
        self.materialName="" #The material to use in the .mtl file.

    def addVertex(self,position,normal):
        newVertexIndex=len(self.vertices)+1
        newVertex=Vertex(position,normal,newVertexIndex)
        self.vertices.append(newVertex)
        return newVertex
      
    #Adds a UV vertex for a particular set of faces associated with a particular real vertex.
    def addUvVertex(self,u,v,realVertex,faces):
        newUvVertexIndex=len(self.uvVertices)+1
        newUvVertex=UvVertex(u,v,newUvVertexIndex)
        self.uvVertices.append(newUvVertex)
        
        for currentFace in faces:
            realVertex.uvVertices[currentFace]=newUvVertex #The UV vertex is associated with the face inside the real vertex.
    
    #Adds a pair of half edges between vertex1 and vertex2 without any attached faces. Returns the halfedge
    #starting at vertex1.
    def addHalfedgePair(self,vertex1,vertex2):
        halfedge1=Halfedge(vertex2)
        halfedge2=Halfedge(vertex1)
        halfedge1.oppositeHalfedge=halfedge2 #Each new halfedge is made the other's opposite.
        halfedge2.oppositeHalfedge=halfedge1
        self.halfedges.append(halfedge1)
        self.halfedges.append(halfedge2)
        vertex1.halfedges.append(halfedge1)
        vertex2.halfedges.append(halfedge2)
        return halfedge1
              
        
    #Creates a face with the vertices in faceVertices. The face winding order is determined by either a detected neighbouring face or by the normal direction of the first vertex
    #if no neighbouring faces exist.
    def addFace(self,faceVertices):                    
        anInitialWindingOrderFace=None #A face existing on at least one of the halfedges initally assigned to the face.
        anOppositeWindingOrderFace=None #A face exisiting on one at least one of the halfedges oppisite to those initally assigned to the face.
        for i2 in range(0,len(faceVertices)): #Loop to find a face already existing on the new vertex loops bounds or on the other side of the bounds. Determines the winding order of the new face.
            i1=i2-1
            vertex1=faceVertices[i1]
            vertex2=faceVertices[i2]
            
            currentHalfedge=vertex1.getHalfedgeTo(vertex2)
            if(currentHalfedge):
                if(currentHalfedge.face):
                    anInitialWindingOrderFace=currentHalfedge.face
                    
                if(currentHalfedge.oppositeHalfedge.face):
                    anOppositeWindingOrderFace=currentHalfedge.oppositeHalfedge.face
        
        
        correctedFaceVertices=faceVertices           
        if(anInitialWindingOrderFace):
            if(anOppositeWindingOrderFace): #The new face cannot be created due to at least two faces it will be neighbouring having different winding orders.
                print("Cannot create new face with vertices "+str([cv.listIndex for cv in faceVertices])+
                      " due to at least two neghbouring faces ("+str([cv.listIndex for cv in anInitialWindingOrderFace.getAllVertices()])+
                      ", "+str([cv.listIndex for cv in anOppositeWindingOrderFace.getAllVertices()])+") having contradicting winding orders.")
                raise BaseException
            else: #The initially assigned halfedges already have at least one face assigned to it so the winding order of the new face needs to be swapped.
                correctedFaceVertices=faceVertices[::-1]
                
            
                   
        #The new face and any new needed halfedges are created below.         
        newFace=None
        for i3 in range(0,len(correctedFaceVertices)): #Loops through all pairs of linked halfedges for the face.
            i2=i3-1
            i1=i3-2
            vertex1=correctedFaceVertices[i1]
            vertex2=correctedFaceVertices[i2]
            vertex3=correctedFaceVertices[i3]
            currentHalfedge1=vertex1.getHalfedgeTo(vertex2)
            currentHalfedge2=vertex2.getHalfedgeTo(vertex3)
            

            #The halfedge pairs from vertex 1 to vertex2 and from vertex2 to vertex3 are created if they don't already exist.
            if(currentHalfedge1 is None):
                currentHalfedge1=self.addHalfedgePair(vertex1,vertex2)
                
            if(currentHalfedge2 is None):
                currentHalfedge2=self.addHalfedgePair(vertex2,vertex3)
                                                                   
                
            currentHalfedge1.nextFaceHalfedge=currentHalfedge2 #The halfedges for the new face are linked together.
            
            if(i3==0): #The face is created on the first iteration of the loop.
                newFace=Face(currentHalfedge1)
                
            currentHalfedge1.face=newFace #The faces's bounding halfedges are linked to the face.
            currentHalfedge2.face=newFace   

            
        newFace.calculateNormal() #The new face's normal is calculated once all it's halfedges have been created.
        self.faces.append(newFace) 

        
    #Splits faceToSplit into two faces split along an edge from v1 to v2. faceToSplit is modified, and a second face is created from the halfedges no longer in faceToSplit and
    #the new splitting halfedge. The new face is returned.
    def splitFace(self,faceToSplit,v1,v2):
        firstHalfedges=faceToSplit.getAllHalfedges()
        centralVertices=faceToSplit.getAllVertices()
        secondHalfedges=list(numpy.roll(firstHalfedges,-1)) #This is firstHalfedges shifted back by 1 element.
        halfedgePairs=zip(firstHalfedges,centralVertices,secondHalfedges)
        
        fheV1,sheV1,fheV2,sheV2=None,None,None,None #The halfedge pairs that are joined to v1 and v2.
        for fhe,cv,she in halfedgePairs: #Loops over all halfedge pairs and the vertices shared by them. Finds the halfedge pairs for this face that have v1 and v2 as their shared vertex.
            if(cv is v1):
                fheV1,sheV1=fhe,she            
            if(cv is v2):
                fheV2,sheV2=fhe,she
   
        splittingHalfedge1=self.addHalfedgePair(v1,v2) #The new halfedge pair that will split the face.
        splittingHalfedge2=splittingHalfedge1.oppositeHalfedge
        
        #The halfedge loop for the already existing face is updated.
        fheV1.nextFaceHalfedge=splittingHalfedge1
        splittingHalfedge1.nextFaceHalfedge=sheV2
        splittingHalfedge1.face=faceToSplit
        faceToSplit.halfedge=splittingHalfedge1
        
        #The halfedge loop for the new face is closed.
        fheV2.nextFaceHalfedge=splittingHalfedge2
        splittingHalfedge2.nextFaceHalfedge=sheV1
        
        #The new face is created and assigned to its halfedges.
        newFace=Face(splittingHalfedge2)
        for currentHalfedge in splittingHalfedge2.getHalfedgeFaceLoop():
            currentHalfedge.face=newFace
            
        newFace.calculateNormal()
        self.faces.append(newFace)        
        return newFace
    
    
    #Recursively triangulates a face using an ear clipping algorithm from "Ear-clipping Based Algorithms of Generating High-quality Polygon Triangulation" 
    #(https://arxiv.org/abs/1212.6038 ,Algorithm 1). It works by recursively removing the ears of a polygon. An ear of a polygon is a triangle formed by two
    #edges of a polygon and a third edge that is completely inside the polygon. The algoroithm removes the ear with the sharpest angle between the two 
    #polygon edges first.
    def triangulateFace(self,faceToTriangulate):
        vertices1,projectedVertices1=faceToTriangulate.getAllVertices(),faceToTriangulate.getNormalPlaneProjection()    
        if(len(vertices1)<4):
            return #A triangle is already fully triangulated.
        
        projectedVertices2=list(numpy.roll(projectedVertices1,-1,axis=0))
        vertices3,projectedVertices3=list(numpy.roll(vertices1,-2)),list(numpy.roll(projectedVertices1,-2,axis=0))
        faceIsClockwise=getAreaOfPolygon(projectedVertices1)>0.0
        
        #Below loops through all corners of the polygon. Each iteration includes the two vertices that are next to the corner, and the 2d projection of the
        #corner and aforementioned neighbouring vertices.
        splittingVertex1,splittingVertex2=None,None #The vertices identified as belonging to the sharpest ear.
        currentSmallestEarAngle=100.0
        internalAngleSum=0.0

        for v1,v3,pv1,pv2,pv3 in zip(vertices1,vertices3,projectedVertices1,projectedVertices2,projectedVertices3):
            pv1X,pv1Y=pv1
            pv2X,pv2Y=pv2
            pv3X,pv3Y=pv3
            
            #Below the internal angle defined by pv1,pv2 and pv3 is determined using a definition of a cross product.
            edgeLength12=(((pv2X-pv1X)**2.0)+((pv2Y-pv1Y)**2.0))**0.5 #These are the two edge lengths.
            edgeLength23=(((pv3X-pv2X)**2.0)+((pv3Y-pv2Y)**2.0))**0.5   
            cornerCrossProduct=((pv2X-pv1X)*(pv3Y-pv2Y))-((pv2Y-pv1Y)*(pv3X-pv2X))
            cornerDotProduct=((pv1X-pv2X)*(pv3X-pv2X))+((pv1Y-pv2Y)*(pv3Y-pv2Y))

            currentCornerAngle=math.acos(max(min(cornerDotProduct/(edgeLength12*edgeLength23),1.0),-1.0)) #The angle between the current pair of edges using the definition of a dot product.
            angleIsReflex=cornerCrossProduct>0.0 if(faceIsClockwise) else cornerCrossProduct<0.0 #The sign of the cross product changes depending on whether the edge vectors represent a reflex angle or not for the current face's winding order.
            currentInternalAngle=(2.0*math.pi)-currentCornerAngle if(angleIsReflex) else currentCornerAngle
            internalAngleSum+=currentInternalAngle

            if(currentInternalAngle>=math.pi):
                continue #The corner is not convex, meaning that is cannot be an ear.
                
            if(currentInternalAngle>=currentSmallestEarAngle):
                continue #This ear has a larger angle than at least one previously found ear.
            
            #Below the current ear candidate is searched for whether other vertices of the face can be found inside it. If they can be, then the triangle is not an ear.
            noPointsInsideEarTriangle=True
            for cpv in projectedVertices1:
                if((cpv==pv1) or (cpv==tuple(pv2)) or (cpv==tuple(pv3))):
                    continue #Vertices that make up the candidate ear are not tested.
                    
                if(pointInsideTriangle(pv1,pv2,pv3,cpv)):
                    noPointsInsideEarTriangle=False #As a vertex of the face has been found inside the triangle meaning that the triangle cannot be an ear.
                    break
                
            if(noPointsInsideEarTriangle): #This triangle is an ear and has a smaller internal angle than previously detected, so it is made the current ear to remove
                splittingVertex1,splittingVertex2=v1,v3
                currentSmallestEarAngle=currentInternalAngle
                
        if(internalAngleSum>(math.pi*(len(vertices1)-2))): #If the sum of internal angles is greater than usual, then this a self-intersecting polygon. It will be split using the first and third vertices in the face's vertex list.
           splittingVertex1,splittingVertex2=vertices1[0],vertices1[2]
               
        #Below the face is split and recursive calls are made to remove more ears. Only one of the recursive calls at most will do anything as at least one is a triangle.
        newFace=self.splitFace(faceToTriangulate,splittingVertex1,splittingVertex2)
        self.triangulateFace(faceToTriangulate)
        self.triangulateFace(newFace)
        
        
    def triangulateAllFaces(self):
        oldFaces=self.faces.copy() #This shallow copy of the list is used because the faces list will change as new faces are added during the triangulation from the splitting of faces.
        
        for currentFace in oldFaces:
            self.triangulateFace(currentFace)
            
        
        
    #Adds a vertex list representing a new face to a list of faces to be added to the mesh. The faces in the list are not created until the createMesh function is called.
    def addFaceToWaitingList(self,faceVertices):
        self.facesToCreate.append(faceVertices)
        
    #Creates a mesh by adding faces in order so that new faces are added to the existing mesh first if the exisitng mesh shares an edge with the new face. This is done in order to
    #ensure the winding order of all added faces is consistent. A face is added in free space only if the existing mesh cannot be expanded by any of the faces currently waiting to be created.
    def createMesh(self):
        while(len(self.facesToCreate)>0): #Loop while there are still faces to be put into the mesh.
            faceFoundToCreate=None
            
            for currentFaceToCreate in self.facesToCreate: #Loops over all faces to still be created.
                if(faceFoundToCreate):
                    break #A face that connects to the exisiting mesh has been found.
                
                for i2 in range(0,len(currentFaceToCreate)): #Loops over all edges in the face to be created.
                    i1=i2-1
                    vertex1=currentFaceToCreate[i1]
                    vertex2=currentFaceToCreate[i2]
                    
                    currentHalfedge=vertex1.getHalfedgeTo(vertex2)
                    if(currentHalfedge): #A halfedge already exists between vertex 1 and vertex 2.
                        if((currentHalfedge.face is None) or (currentHalfedge.oppositeHalfedge.face is None)): #If at least one face along the halfedge pair is unoccupied.
                            faceFoundToCreate=currentFaceToCreate
                            
                            
            if(faceFoundToCreate): #The face is added to the mesh and removed from the waiting list.
                self.addFace(faceFoundToCreate)
                self.facesToCreate.remove(faceFoundToCreate)
            else: #None of the waiting faces join to any part of the existing mesh, meaning that the remaining faces form at least one disconnected region. It is therefore safe to add the first face in the waiting list to free space.
                firstFaceToCreate=self.facesToCreate[0]
                self.addFace(firstFaceToCreate)
                self.facesToCreate.remove(firstFaceToCreate)
                            
                
 
            
    #Writes the data for the current mesh to an .obj file as an object. In order to accomodate multiple mesh objects in a single .obj file there are vertex index offset arguments.  
    def writeToObjFile(self,objFile,vertexPositionIndexOffset,vertexUvIndexOffset):
        if(self.mtlFilename!=""):
            objFile.write("mtllib "+self.mtlFilename+"\n") #The path to the material file.           
        objFile.write("o "+self.name+"\n")
        meshIsTextured=len(self.uvVertices)>0

        for currentVertex in self.vertices: #Writes vertex position data.
            vertexPositionString="v "+str(currentVertex.position[0])+" "+str(currentVertex.position[1])+" "+str(currentVertex.position[2])+"\n"
            objFile.write(vertexPositionString)
            
        if(meshIsTextured): #Writes the UV vertex data if needed.
            for currentUvVertex in self.uvVertices:
                uvVertexString="vt "+str(currentUvVertex.position[0])+" "+str(currentUvVertex.position[1])+"\n"
                objFile.write(uvVertexString)
                
        for currentVertex in self.vertices: #Writes the vertex normal data.
            vertexNormalString="vn "+str(currentVertex.normal[0])+" "+str(currentVertex.normal[1])+" "+str(currentVertex.normal[2])+"\n"
            objFile.write(vertexNormalString)
        
        if(self.mtlFilename!=""):
            objFile.write("usemtl "+self.materialName+"\n") #The material name for this mesh in the material file.
        objFile.write("s on\n") #Turns on smooth shading which interpolates the normals across each face accoding to the normals at the face edges.
        for i,currentFace in enumerate(self.faces):
            currentFaceVertices=currentFace.getAllVertices()
            if(not currentFace.windingOrderAndNormalAligned()): #The winding order written to the .obj file is swapped if needed to make sure that the face is oriented according to the face's normal.
                currentFaceVertices=currentFaceVertices[::-1]

            currentFaceVertexIndices=[str(currentVertex.listIndex+vertexPositionIndexOffset) for currentVertex in currentFaceVertices] #The vertex list indices for the current face.
                       
            currentFaceUvVertexIndices=[str("") for i in range(0,len(currentFaceVertexIndices))]
            if(meshIsTextured):
                currentFaceUvVertices=currentFace.getAllUvVertices()
                currentFaceUvVertexIndices=[str(currentUvVertex.listIndex+vertexUvIndexOffset) for currentUvVertex in currentFaceUvVertices]
            
            faceString="f"
            for currentFaceVertexIndex,currentFaceUvVertexIndex in zip(currentFaceVertexIndices,currentFaceUvVertexIndices): #Adds the position,uv and normal list indices for each vertex in the face.
                faceString+=" "
                faceString+=currentFaceVertexIndex+"/"+currentFaceUvVertexIndex+"/"+currentFaceVertexIndex  
            faceString+="\n"
            
            objFile.write(faceString)
            
            
            
#Writes a list of meshes to a single .obj file.           
def writeMeshesToObjFile(fileName,meshes):
    objFile=open(fileName+".obj","wt")    
    vertexPositionIndexOffset=0
    vertexUvIndexOffset=0
    
    for currentMesh in meshes:
        currentMesh.writeToObjFile(objFile,vertexPositionIndexOffset,vertexUvIndexOffset)
        
        #The vertex indices for the next mesh are advanced by the number of vertices in the current mesh.
        vertexPositionIndexOffset+=len(currentMesh.vertices)
        vertexUvIndexOffset+=len(currentMesh.uvVertices)
        
    objFile.close()
    
