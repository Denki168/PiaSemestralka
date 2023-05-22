// Gmsh project created on Fri Apr 16 11:47:31 2021
SetFactory("OpenCASCADE");

//+
Point(1) = {0, 0, 0, 0.03};
//+
Point(2) = {0.1666, 0, 0, 0.03};
//+
Point(3) = {0.1666, 1, 0, 0.03};
//+
Point(4) = {0, 1, 0, 0.03};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve(1) = {4, 3, 2, 1};
//+
Physical Surface(2) = {1};
