SetFactory("OpenCASCADE");

Merge "../../model/gelsight_mini.step";


//======================================================================================================
//.....Meshing options...... (only general options are set here) 
//======================================================================================================

//set 2D mesh algorithm (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 
// 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms, 
// 11: Quasi-structured Quad)
Mesh.Algorithm=6;

//set 3D mesh algorithm (1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT)
Mesh.Algorithm3D=1;

//set recombination algorithm (0: simple, 1: blossom, 2: simple full-quad, 3: blossom full-quad)
//and recombine all surfaces
Mesh.RecombinationAlgorithm=1;
Mesh.RecombineAll=0;

//set subdivision algorithm (0: none, 1: all quadrangles, 2: all hexahedra, 3: barycentric)
Mesh.SubdivisionAlgorithm=0;

//smoothing steps
Mesh.Smoothing=22;

//element size factor
Mesh.MeshSizeFactor=1;

//min/max element sizes
Mesh.MeshSizeMin=0;
Mesh.MeshSizeMax=1e22;

//element order
Mesh.ElementOrder=2;

//use incomplete elements (CALCULIX cannot handle C3D27 elements !!!)
Mesh.SecondOrderIncomplete=0;

//======================================================================================================
//.....Mesh properties......
//======================================================================================================


//Add size field (contact surface should be finer)
Field[1] = Box;
Field[1].XMax = 13;
Field[1].XMin = -13;
Field[1].YMax = 23.5;
Field[1].YMin = 22;
Field[1].ZMax = 10;
Field[1].ZMin = -11;
Background Field = 1;
Field[1].VOut = 1.2;
Field[1].VIn = 0.75;
Field[1].Thickness = 0.75;

//add physical volume to not export surface elements
Physical Volume("gelsight_miniVolume", 1) = {1};

//==============
// ...MESHING...
//==============

Mesh 1;
Mesh 2;
Mesh 3;
