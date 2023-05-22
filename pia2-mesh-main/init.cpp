#include "init.h"
#include "MeshGmsh.h"

#define _USE_MATH_DEFINES
#include <cmath>

void initVazDesk(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds){
	m = new MeshGmsh("VazDeska.msh");
	W = new Field<Compressible>(*m);
	boundary_conds.push_back(new SlipWallBC({11}));  //dolni vlevo
	boundary_conds.push_back(new vazDeskBC({12})); //dolni vpravo jina podminka
	boundary_conds.push_back(new puBC({13})); //pravo
	boundary_conds.push_back(new puBC({14})); //pravo
	boundary_conds.push_back(new SlipWallBC({15})); //nahore
	boundary_conds.push_back(new muBC({16}));	//vlevo
	boundary_conds.push_back(new muBC({17}));	//vlevo

    //pocatecni podminky
	for (int i=0; i<m->nc; ++i) {
		Polygon const& T_x = m->cell[i];
		double rho, e, p;
		Vector2D u;

        rho = 1.0201201598403826;
        u = {0.0,0.0};
        p = 18.0;

        (*W)[i].rho = rho;
        (*W)[i].rhoU = rho*u;
        (*W)[i].e = (*W)[i].eos_e_from_p(p);

	}

	for (auto bc : boundary_conds) bc->apply(*m,*W);
}
void initSod(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds) {
	int nx = 1000;
	m = new Mesh(0.,1.,0.,1./double(nx),nx,1);
	//m = new MeshGmsh("square.msh");
	W = new Field<Compressible>(*m);
	boundary_conds.push_back(new SlipWallBC({1}));

	for (int i=0; i<m->nc; ++i) {
		Polygon const& T_x = m->cell[i];
		double rho, e, p;
		Vector2D u;
		if(T_x.centroid().x < 0.5){
			rho = 1.0;
			u = {0.0,0.0};
			p = 1.0;
		}
		else{
			rho = 0.125;
			u = {0.0,0.0};
			p = 0.1;
		}
			(*W)[i].rho = rho;
			(*W)[i].rhoU = rho*u;
			(*W)[i].e = (*W)[i].eos_e_from_p(p);
	}

	for (auto bc : boundary_conds) bc->apply(*m,*W);
}

void initJet(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds) {
	m = new MeshGmsh("jet.msh");
	W = new Field<Compressible>(*m);
	boundary_conds.push_back(new SlipWallBC({1}));
	boundary_conds.push_back(new ReservoirBC({2}));

	for (int i=0; i<m->nc; ++i) {
		Polygon const& T_x = m->cell[i];

			(*W)[i].rho = 1.0;
			(*W)[i].rhoU = Vector2D(0.0,0.0);
			(*W)[i].e = (*W)[i].eos_e_from_p(0.1);
	}

	for (auto bc : boundary_conds) bc->apply(*m,*W);
}

void initKH(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds) {
	m = new MeshGmsh("KelvinHelmholtz_3x05.msh");
	W = new Field<Compressible>(*m);
	boundary_conds.push_back(new SlipWallBC({9}));  //horni
	boundary_conds.push_back(new SlipWallBC({10})); //dolni
	boundary_conds.push_back(new ConstVelocityBC({11},{0.5,0.0})); //levo dole
	boundary_conds.push_back(new ConstVelocityBC({12},{-0.5,0.0})); //pravo hore
	boundary_conds.push_back(new ConstVelocityBC({13},{0.5,0.0})); //pravo dole
	boundary_conds.push_back(new ConstVelocityBC({14},{-0.5,0.0}));	//levo hore


	for (int i=0; i<m->nc; ++i) {
		Polygon const& T = m->cell[i];
		double rho, e, p;
		Vector2D u;
		if(T.centroid().y < 0.25){
			rho = 1.0;
			u = {0.5,0.0};
			p = 2.5;
		}
		else{
			rho = 2.0;
			u = {-0.5,0.0};
			p = 2.5;
		}
			(*W)[i].rho = rho;
			(*W)[i].rhoU = rho*u;
			(*W)[i].e = (*W)[i].eos_e_from_p(p);
	}

	for (auto bc : boundary_conds) bc->apply(*m,*W);
}

void initRayTay(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds, int reg) {
	Compressible::g = 0.1;

	switch(reg){
		case 1:
			m = new MeshGmsh("RayTay_sparse.msh");
			break;
		case 0:
			m = new MeshGmsh("RayTay_dense.msh");
			break;
		case 2:
			m = new MeshGmsh("RayTay_double_dense.msh");
			break;
		default:
			std::cout << "*.msh file not found.";
	}

	W = new Field<Compressible>(*m);
	boundary_conds.push_back(new SlipWallBC({1}));
	double yPol, p, rho1=2.0, rho2=1.0, g=0.1;
	for (int i=0; i<m->nc; ++i) {
		Polygon const& T_x = m->cell[i];
	    yPol = T_x.centroid().y;
		if (yPol > 0.5){
			(*W)[i].rho = 2.0;
			(*W)[i].rhoU = Vector2D(0.0,0.0);
			p = 1.0 + (1.0-yPol) * rho1 * g;
			(*W)[i].e = (*W)[i].eos_e_from_p(p);
		}else{
			(*W)[i].rho = 1.0;
			(*W)[i].rhoU = Vector2D(0.0,0.0);
			p = 1.0 + (0.5*rho1+(1.0-yPol) * rho2)*g;
			(*W)[i].e = (*W)[i].eos_e_from_p(p);
		}


	}

	for (auto bc : boundary_conds) bc->apply(*m,*W);
}

void initRayTayCos(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds, int reg) {
	Compressible::g = 0.1;

	switch(reg){
		case 1:
			m = new MeshGmsh("RayTay_sparse.msh");
			break;
		case 0:
			m = new MeshGmsh("RayTay_dense.msh");
			break;
		case 2:
			m = new MeshGmsh("RayTay_double_dense.msh");
			break;
		default:
			std::cout << "*.msh file not found.";
	}

	W = new Field<Compressible>(*m);
	boundary_conds.push_back(new SlipWallBC({1}));
	if(reg==3)boundary_conds.push_back(new ReservoirBC({2}));
	double yPol, xPol, yLim, p, rho1=2.0, rho2=1.0, g=0.1;
	double a=0.05, c=0.5, b=2.0*M_PI/0.1666;
	for (int i=0; i<m->nc; ++i) {
		Polygon const& T_x = m->cell[i];
	    yPol = T_x.centroid().y;
	    xPol = T_x.centroid().x;
	    yLim = a*cos(b*xPol)+c;
		if (yPol > yLim){
			(*W)[i].rho = 2.0;
			(*W)[i].rhoU = Vector2D(0.0,0.0);
			p = 1.0 + (1.0-yPol) * rho1 * g;
			(*W)[i].e = (*W)[i].eos_e_from_p(p);
		}else{
			(*W)[i].rho = 1.0;
			(*W)[i].rhoU = Vector2D(0.0,0.0);
			p = 1.0 + (0.5*rho1+(1.0-yPol) * rho2)*g;
			(*W)[i].e = (*W)[i].eos_e_from_p(p);
		}
	}

	for (auto bc : boundary_conds) bc->apply(*m,*W);
}
