// Gmsh project created on Tue May 17 10:28:50 2022
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {2, 0, 0, 0.1};
Point(4) = {2, 2, 0, 0.1};
Point(5) = {0, 2, 0, 0.1};
Point(6) = {0, 0.25, 0, 0.1};
Point(7) = {2, 0.25, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 7};
Line(4) = {7, 4};
Line(5) = {4, 5};
Line(6) = {5, 6};	
Line(7) = {6, 1};
Line(8) = {6, 7};
//Line Loop(9) = {7, 6, 5, 1, 2, 3, 4};
//Plane Surface(10) = {10};
Physical Line(11) = {1};
Physical Line(12) = {2};
Physical Line(13) = {3};
Physical Line(14) = {4};
Physical Line(15) = {5};
Physical Line(16) = {6};
Physical Line(17) = {7};//+
Curve Loop(1) = {5, 6, 7, 1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Surface(18) = {1};
//+
MeshSize {1, 2} = 0.005;
//+
MeshSize {2, 3} = 0.0025;
//+
MeshSize {3, 7} = 0.0025;
//+
MeshSize {6, 1} = 0.05;
//+
MeshSize {5, 4} = 0.3;
