# Phantom-SPH-isosurface-model-generator
Creates 3D isosurface models from Phantom SPH simulations.

Currently only creates isosurfaces of density. Uses Phantom .h5 files (Phantom SPH link: https://phantomsph.bitbucket.io/#home). Uses the technique "Dual Contouring of Hermite Data" (https://www.cse.wustl.edu/~taoju/research/dualContour.pdf) to create isosurfaces. Isosurface models are outputted in the .obj format. Sink masses are modelled as small spheres as separate objects in the same .obj file as the isosurface.

The program also creates two equirectangular projected velocity maps that can be used as a texture for the isosurface model (the model has an equirectngular UV map); the velocity projected in the surface normal direction (corresponding to expansion and contraction) and the velocity projected in the direction perpendicular to both the Z axis and the surface normal (corresponds to rotation on the XY plane).

