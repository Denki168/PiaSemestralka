#include "BC.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#define kappa 1.4

void SlipWallBC::apply(Mesh const& m, Field<Compressible> & W) {
	for (auto const& e : m.edge) {
		if (isCorrectLocation(e.location)) {
			int cl = e.left();  // Internal cell index
			int cr = e.right(); // Ghost cell index

			Vector2D en = e.unitNormal();
			Vector2D un = en*dot(W[cl].u(),en); // Normal component of the fluid velocity relative to the edge
			Vector2D ur = W[cl].u() - 2.0*un; // Velocity vector mirrored along the edge

			W[cr].rho = W[cl].rho;
			W[cr].rhoU = W[cl].rho * ur;
			W[cr].e = W[cl].e;
		}
	}
}
void vazDeskBC::apply(Mesh const& m, Field<Compressible> & W) {
	for (auto const& e : m.edge) {
		if (isCorrectLocation(e.location)) {
            int cl = e.left();  // Internal cell index
			int cr = e.right(); // Ghost cell index

			Vector2D ur = -1.0 * W[cl].u(); // Velocity vector is zero along the perfect viscous wall

			W[cr].rho = W[cl].rho;
			W[cr].rhoU = W[cl].rho * ur;
			W[cr].e = W[cl].e;

		}
	}
}
void ReservoirBC::apply(Mesh const& m, Field<Compressible> & W) {
	for (auto const& e : m.edge) {
		if (isCorrectLocation(e.location)) {
			int cl = e.left();  // Internal cell index
			int cr = e.right(); // Ghost cell index

			W[cr].rho = 1.0;
			W[cr].rhoU = Vector2D(0.0,0.0);
			W[cr].e = W[cr].eos_e_from_p(1);
		}
	}
}

void FreeBC::apply(Mesh const& m, Field<Compressible> & W) {
	for (auto const& e : m.edge) {
		if (isCorrectLocation(e.location)) {
			int cl = e.left();  // Internal cell index
			int cr = e.right(); // Ghost cell index

			W[cr].rho = W[cl].rho;
			W[cr].rhoU = W[cl].rhoU;
			W[cr].e = W[cl].e;
		}
	}
}

void ConstVelocityBC::apply(Mesh const& m, Field<Compressible> & W) {
	for (auto const& e : m.edge) {
		if (isCorrectLocation(e.location)) {
			int cl = e.left();  // Internal cell index
			int cr = e.right(); // Ghost cell index

			W[cr].rho = W[cl].rho;
			W[cr].rhoU = W[cl].rho*velocity;
			W[cr].e = W[cl].e;
		}
	}
}

void puBC::apply(Mesh const& m, Field<Compressible> & W) {
	for (auto const& e : m.edge) {
		if (isCorrectLocation(e.location)) {
			int cl = e.left();  // Internal cell index
			int cr = e.right(); // Ghost cell index
			double pout = 17.8571428571428571;

			W[cr].rho = W[cl].rho;
			W[cr].rhoU = W[cl].rho*Vector2D(0.5,0.0);
			W[cr].e = W[cl].e;
		}
	}
}

void muBC::apply(Mesh const& m, Field<Compressible> & W) {
	for (auto const& e : m.edge) {
		if (isCorrectLocation(e.location)) {
			int cl = e.left();  // Internal cell index
			int cr = e.right(); // Ghost cell index

			double p0in = 18.362162877126892;
			//double pin = 18.362162877126892;
            double pl = W[cl].p();
            double pin = std::min(p0in,pl);
			double c = kappa*(pin/p0in);
			double M = (2/(kappa-1))*(pow ((pin/p0in),-((kappa-1/kappa)) ) - 1);
            double rhoin = 1.0201201598403826 * pow((pin/p0in),(1/kappa));
			W[cr].rho =  rhoin;
			W[cr].rhoU = rhoin * sqrt(M * c) * Vector2D(-1,0.0);
			W[cr].e = pin / (kappa - 1.0) + 0.5 * rhoin * sqrt(M * c) ;

		}
	}
}

