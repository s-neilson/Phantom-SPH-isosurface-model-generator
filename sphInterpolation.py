import math
import numpy



#Returns the gaussian kernal function for SPH for every position in rFroms, from page 3 of "Smoothed Particle Hydrodynamics and Magnetohydrodynamics"
#by Daniel J Price, https://arxiv.org/abs/1012.1885
def W(h,rFroms,rTo):
    rDifferences=rTo-rFroms
    separationsSquared=numpy.power(numpy.linalg.norm(rDifferences,axis=1),2.0)

    constantFactor=1.0/(math.pi**(3.0/2.0))
    hFactors=numpy.expand_dims(1.0/(numpy.power(h,3.0)),axis=1)
    expodents=numpy.expand_dims(((-1.0)/numpy.power(h,2.0))*separationsSquared,axis=1)
    return constantFactor*hFactors*numpy.exp(expodents)

#Returns the gradient of the above gaussian kernal function for every position in rFroms.
def dW(h,rFroms,rTo):
    rDifferences=rTo-rFroms
    constantFactors=numpy.expand_dims((-2.0)/(numpy.power(h,2.0)),axis=1)
    return rDifferences*constantFactors*W(h,rFroms,rTo)




#Performs SPH interpolation for a value over positions in a list
def interpolatePositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleValues):
    interpolatedValues=numpy.zeros(shape=(numpy.array(samplingPositions).shape[0],1))

    closestParticleIndices=particleTree.query(samplingPositions,1,return_distance=False) #The indices of the particles closest to each sampling point.
    closestParticleH=particleH[closestParticleIndices.squeeze()] #The smoothing lengths of the SPH particles closest to each sampling point.
    neighbouringParticleIndices=particleTree.query_radius(samplingPositions,2.0*closestParticleH) #For each sampling point, the neighbouring SPH particle indices within twice the nearest particle smoothing length.
    
    
    for i in range(0,len(samplingPositions)): #Loops over all sampling positions.
        #The coordinates of the point to be sampled.
        currentSamplingPosition=samplingPositions[i] #The coordinates of the point that will have a value interpolated.

        currentNeighbouringParticleIndices=neighbouringParticleIndices[i] #The neighbouring SPH particle indices for the current sampling point.
        neighbouringParticleH=particleH[currentNeighbouringParticleIndices]
        neighbouringParticlePositions=particlePositions[currentNeighbouringParticleIndices]
        neighbouringParticleDensities=particleDensities[currentNeighbouringParticleIndices]
        neighbouringParticleValues=numpy.expand_dims(particleValues[currentNeighbouringParticleIndices],axis=1)
        
        wValues=W(neighbouringParticleH,neighbouringParticlePositions,currentSamplingPosition) #An array containing the W contributions from every neighbouring particle is created.       
        particleContributions=wValues*neighbouringParticleValues*(particleMass/neighbouringParticleDensities) #An array containing the contribution of each neighbouring particle to the total interpolated value.

        interpolatedValues[i,0]=particleContributions.sum()
                
    return interpolatedValues

#Performs SPH interpolation for a value over a 3d grid of positions.
def interpolateGridPositions(samplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleValues):
    numberOfSamples=numpy.prod(samplingPositions.shape[0:3])
    flattenedSamplingPositions=samplingPositions.reshape((numberOfSamples,3),order="C") #Turns the 3d array of sample position vectors into a 1d array of sample position vectors.

    flattenedInterpolatedValues=interpolatePositions(flattenedSamplingPositions,particleTree,particleMass,particleH,particlePositions,particleDensities,particleValues)
    interpolatedValues=flattenedInterpolatedValues.reshape(samplingPositions.shape[0:3]+(1,),order="C") #The interpolated values for each position are put into a 3d array with indices corresponding to the original sample positions in samplingPositions.
     
    return interpolatedValues
                
                
    
    