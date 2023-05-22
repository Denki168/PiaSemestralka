#include "output.h"
#include <fstream>

void outputVTK(std::string filename, Mesh const& m, Field<double> const& u) {
	std::ofstream f;
	f.open(filename,std::ios::out);
	
	f << "# vtk DataFile Version 1.0\n";
	f << "unstructured grid\n";
	f << "ascii\n";
	f << "DATASET UNSTRUCTURED_GRID\n";
	
	f << "points " << m.node.size() << " float\n";
	
	for (int i=0; i<m.node.size(); ++i) {
		f << m.node[i] << "\n";
	}
	
	f << "cells " << m.nc << " " << m.nCellNodes() + m.nc << "\n";
	
	for (int i=0; i<m.nc; ++i) {
		f << m.cell[i] << "\n";
	}
	
	f << "cell_types " << m.nc << "\n";
	
	for (int i=0; i<m.nc; ++i) {
		f << "9\n";
	}
	
	f << "CELL_DATA " << m.nc << "\n";
 	f << "SCALARS u float\n"; 
	f << "LOOKUP_TABLE default\n";
	
    for (int i=0; i<m.nc; ++i) {
		f << u[i] << "\n";
	}
	
	f.close();
}

void outputVTK(std::string filename, Mesh const& m, Field<Compressible> const& u) {
	std::ofstream f;
	f.open(filename,std::ios::out);
	
	f << "# vtk DataFile Version 1.0\n";
	f << "unstructured grid\n";
	f << "ascii\n";
	f << "DATASET UNSTRUCTURED_GRID\n";
	
	f << "points " << m.node.size() << " float\n";
	
	for (int i=0; i<m.node.size(); ++i) {
		f << m.node[i] << "\n";
	}
	
	f << "cells " << m.nc << " " << m.nCellNodes() + m.nc << "\n"; // TODO: upravit az bude funkce nCellNodes
	
	for (int i=0; i<m.nc; ++i) {
		f << m.cell[i] << "\n";
	}
	
	f << "cell_types " << m.nc << "\n";
	
	for (int i=0; i<m.nc; ++i) {
		f << "9\n";
	}
	
	f << "CELL_DATA " << m.nc << "\n";
 	f << "SCALARS rho float\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].rho << "\n";
	}

 	//f << "SCALARS |u| float\n"; 
	//f << "LOOKUP_TABLE default\n";

    //for (int i=0; i<m.nc; ++i) {
	//	f << u[i].u().norm() << "\n";
	//}		

	//f << "VECTORS u float\n"; 
 	f << "SCALARS u float 3\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].u().x << " " << u[i].u().y << " 0" << "\n";
	}	

 	f << "SCALARS e float\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].epsilon() << "\n";
	}
	
	f << "SCALARS p float\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].p() << "\n";
	}
	
 	f << "SCALARS x float 3\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << m.cell[i].centroid() << "\n";
	}	

	f.close();
}

void outputVTKTimeStep(double time, Mesh const& m, Field<Compressible> const& u) {
	std::ofstream f;
	std::string filename;
	
	filename = "T" + std::to_string(int(time*1000)) + ".vtk";
	f.open(filename,std::ios::out);
	
	f << "# vtk DataFile Version 1.0\n";
	f << "unstructured grid\n";
	f << "ascii\n";
	f << "DATASET UNSTRUCTURED_GRID\n";
	
	f << "points " << m.node.size() << " float\n";
	
	for (int i=0; i<m.node.size(); ++i) {
		f << m.node[i] << "\n";
	}
	
	f << "cells " << m.nc << " " << m.nCellNodes() + m.nc << "\n"; // TODO: upravit az bude funkce nCellNodes
	
	for (int i=0; i<m.nc; ++i) {
		f << m.cell[i] << "\n";
	}
	
	f << "cell_types " << m.nc << "\n";
	
	for (int i=0; i<m.nc; ++i) {
		f << "9\n";
	}
	
	f << "CELL_DATA " << m.nc << "\n";
 	f << "SCALARS rho float\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].rho << "\n";
	}

 	//f << "SCALARS |u| float\n"; 
	//f << "LOOKUP_TABLE default\n";

    //for (int i=0; i<m.nc; ++i) {
	//	f << u[i].u().norm() << "\n";
	//}		

	//f << "VECTORS u float\n"; 
 	f << "SCALARS u float 3\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].u().x << " " << u[i].u().y << " 0" << "\n";
	}	

 	f << "SCALARS e float\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].epsilon() << "\n";
	}
	
	f << "SCALARS p float\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << u[i].p() << "\n";
	}
	
 	f << "SCALARS x float 3\n"; 
	f << "LOOKUP_TABLE default\n";

    for (int i=0; i<m.nc; ++i) {
		f << m.cell[i].centroid() << "\n";
	}	

	f.close();
}
