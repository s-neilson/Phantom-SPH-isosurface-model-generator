import numpy
from sphInterpolation import interpolatePositions
from sphInterpolation import interpolateGridPositions

#Calculated the densities of PHANTOM SPH particles from their mass, smoothing length and smoothing length factor. Uses equation 6 in the
#PHANTOM paper https://users.monash.edu/~dprice/pubs/preprints/phantom-accepted.pdf
def calculateSphParticleDensities(particleMass,particleH,particleHFactor):
    return numpy.expand_dims(particleMass*numpy.power(particleHFactor/particleH,3.0),axis=1)

def interpolateGrid_density(samplingGridPositions,particleTree,particleMass,particleH,particlePositions):
    densities=interpolateGridPositions(samplingGridPositions,particleTree,particleMass,particleH,particlePositions,numpy.ones(shape=(particleH.shape[0],1)),numpy.ones(shape=particleH.shape))
    return densities


#The velocity projected on the unit vector perpendicular to the Z axis and the sampling position normal.
def interpolateList_vNormalXY(samplingPositions,samplingPositionNormals,particleTree,particleMass,particleH,particlePositions,particleVelocities,particleHFactor): 
    particleDensities=calculateSphParticleDensities(particleMass,particleH,particleHFactor)
    vX=interpolatePositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleVelocities[:,0])
    vY=interpolatePositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleVelocities[:,1])
    vZ=interpolatePositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleVelocities[:,2])
    vXYZ=numpy.column_stack((vX,vY,vZ)) #A 2d array is created with the first axis being the sample and the second axis the velocity components.
    
    #The unit vectors perpendicular to the Z axis and the sampling position normals.
    tangentXY=numpy.cross([0.0,0.0,1.0],numpy.array(samplingPositionNormals))
    tangentXY/=numpy.expand_dims(numpy.linalg.norm(tangentXY,axis=1),axis=1)
    vXY=numpy.sum(vXYZ*tangentXY,axis=1) #The dot product between the velocities and the above unit vector.
    return vXY


#The velocity projected on the sampling point normal vector.
def interpolateList_vNormalR(samplingPositions,samplingPositionNormals,particleTree,particleMass,particleH,particlePositions,particleVelocities,particleHFactor):
    particleDensities=calculateSphParticleDensities(particleMass,particleH,particleHFactor)
    vX=interpolatePositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleVelocities[:,0])
    vY=interpolatePositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleVelocities[:,1])
    vZ=interpolatePositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleVelocities[:,2])
    vXYZ=numpy.column_stack((vX,vY,vZ)) #A 2d array is created with the first axis being the sample and the second axis the velocity components.
       
    vR=numpy.sum(vXYZ*numpy.array(samplingPositionNormals),axis=1) #The dot product between the velocities and the sampling point normals.
    return vR