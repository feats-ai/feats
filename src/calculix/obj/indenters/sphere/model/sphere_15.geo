SetFactory("OpenCASCADE");

// we not gonna merge a step file, we construct the sphere directly

// create sphere 
// Points
Point(2) = {0, 0, 0, 1.0};
Point(1) = {7.5, 0, 0, 1.0};
Point(3) = {-7.5, 0, 0, 1.0};

// Lines
Circle(1) = {1, 2, 3};
Line(2) = {3,1};

// Surface
Curve Loop(1) = {1, 2};
Plane Surface(1) = {1};

// create hemisphere (shutdown layers for tetrahedral mesh)
Extrude {{1, 0, 0}, {0, 0, 0}, -Pi} {
  Surface{1}; //Layers{10}; Recombine;
}

// add physical volume for export
Physical Volume("SphereVolume", 6) = {1};
