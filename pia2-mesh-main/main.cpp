#include "Mesh.h"
#include "Field.h"
#include "Compressible.h"
#include "init.h"
#include "BC.h"
#include "output.h"
#include <fenv.h>
#include <iostream>
#include <cstdlib>
#include <string>

int main(int iargc, char* iargv[]) {
	
    Mesh *m;
    Field<Compressible> *W;
    std::vector<BC<Compressible>*> boundary_conds; // Array of boundary conditions
    
    feenableexcept(FE_DIVBYZERO || FE_INVALID || FE_OVERFLOW);

	//initJet(m,W,boundary_conds);
	//initRayTayCos(m,W,boundary_conds,1);//1-sparse, 0-dense, 2-double dense
    
	//initSod(m,W,boundary_conds);
    //const double t_max = 0.8, tSavePer = 0.2;
	//initVazDesk(m,W,boundary_conds);
	initKH(m,W,boundary_conds);
	const double t_max = 0.5, tSavePer = 0.1; //1.0
	
	double dt, t = 0.0;
	int n = 0;
		
	outputVTK("output_init.vtk",*m,*W);
	outputVTKTimeStep(t,*m,*W);
	
	while (t < t_max) {
		dt = timestep(*m,*W);
		
		FVMstep(*m,*W,dt);
		
		for (auto bc : boundary_conds) bc->apply(*m,*W);
		
		n++;
		t += dt;
		if(n%20 == 0) std::cout << roundf(t*100./t_max) <<"% Step " << n << ", time = " << t << "/" << t_max << "" ", dt = " << dt << "\n";
		
		if(int(t*1000)%int(tSavePer*1000) == 0) outputVTKTimeStep(t,*m,*W);
		
	}

	outputVTK("output.vtk",*m,*W);

	delete m; delete W;
	for (auto bc : boundary_conds) delete bc;
    	
	return 0;
}

