// Gmsh project created on Mon Apr  5 23:33:10 2021
Point(1) = {0, 0, 0, 0.01};
Point(2) = {1, 0, 0, 0.01};
Point(3) = {1, 1, 0, 0.01};
Point(4) = {0, 1, 0, 0.01};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};
Physical Surface(7) = {6};
Physical Line(1) = {4, 1, 2, 3}; // This numbering is used for boundary conditions
