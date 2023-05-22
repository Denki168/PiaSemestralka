#include <omp.h>
#include "Compressible.h"
#include "Vector2D.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <float.h>

double Compressible::g = 0.0;

double Compressible::epsilon() const
{
	Vector2D u = rhoU / rho;
	double uSq = dot(u, u);
	return e / rho - 0.5 * uSq;
}
double Compressible::p() const
{
	return (kappa - 1) * rho * epsilon();
}
double Compressible::c() const
{
	return std::sqrt(kappa * p() / rho);
}
Vector2D Compressible::u() const
{
	return rhoU / rho;
}
double Compressible::eos_e_from_p(double p) const
{
	return p / (kappa - 1.0) + 0.5 * rho * dot(u(), u());
}

Compressible fluxUpwind(Compressible Wl, Compressible Wr, Vector2D ne)
{
	double g = 0.1;
	Compressible F = Wl;

	Vector2D ul = Wl.rhoU / Wl.rho;
	Vector2D ur = Wr.rhoU / Wr.rho;

	Vector2D ue = 0.5 * (ul + ur);

	if (dot(ue, ne) < 0)
	{
		F = Wr;
	}

	F = F * dot(ue, ne);

	double pe = 0.5 * (Wl.p() + Wr.p());

	F.rhoU = F.rhoU + pe * ne;
	F.e = F.e + pe * dot(ue, ne);

	// F.rhoU.y+=F.rho*g;

	return F;
}

Vector2D GradU(double Uc, Mesh const &m, int p, Field<Compressible> const &f, int decision)  // Funkce pro vypocet gradientu velicin rho, U , epsilon 
{																							 // pomoci metody nejmensich ctvrecu, 
	double a11;																				
	double a12;
	double a21;
	double a22;

	double b1;
	double b2;

	auto pol = m.cell[p];

	double xc = pol.centroid().x; // Souradnice stredu bunek
	double yc = pol.centroid().y;

	std::vector<int> nghbr; 	 

	for (auto n : pol.node_id)
	{
		for (auto c : m.node[n].cell_ngbr)
		{
			nghbr.push_back(c);				
		}
	}

	std::sort(nghbr.begin(), nghbr.end());
	std::unique(nghbr.begin(), nghbr.end());	// vektor sousednych bunek	

	for (auto b : nghbr)
	{
		double xi = m.cell[b].centroid().x;		// souradnice stredu okolitych stredu bunek
		double yi = m.cell[b].centroid().y;

		double Ui;								// hodnoty velicin v stredoch susednch bunek
		if (decision == 1)
		{
			Ui = f[b].rho;
		}
		else if (decision == 2)
		{
			Ui = f[b].rhoU.x;
		}
		else if (decision == 3)
		{
			Ui = f[b].rhoU.y;
		}
		else if (decision == 4)
		{
			Ui = f[b].epsilon();
		}

		a11 += (xi - xc) * (xi - xc);			// dopocet hodnot matice
		a12 += (xi - xc) * (yi - yc);
		a21 += (xi - xc) * (yi - yc);
		a22 += (yi - yc) * (yi - yc);

		b1 += (Ui - Uc) * (xi - xc);			// dopocet hodnot na prave strane
		b2 += (Ui - Uc) * (yi - yc);
	}
	double A = (a11 * a22) - (a12 * a21);
	double A1 = ((b1 * a22) - (b2 * a12));
	double A2 = ((b2 * a11) - (b1 * a21));
	double Ux = 0;
	double Uy = 0;
	if (A != 0)								// reseni matice pomoci Cramerova pravidla
	{
		double Ux = A1 / A;
		double Uy = A2 / A;
	}

	return Vector2D(Ux, Uy);
}

double Psi(double Uc, Mesh const &m, int p, Field<Compressible> const &f, int decision)  // funkce pro vypocet Barth-Jespersenova limiteru
{

	auto pol = m.cell[p];						// nacteni parametru potrebnych pro vypocet

	double xc = pol.centroid().x;
	double yc = pol.centroid().y;
	Vector2D Xc(xc, yc);

	std::vector<int> nghbr;

	for (auto n : pol.node_id)
	{
		for (auto c : m.node[n].cell_ngbr)
		{
			nghbr.push_back(c);
		}
	}

	std::sort(nghbr.begin(), nghbr.end());
	std::unique(nghbr.begin(), nghbr.end());

	double Umax = DBL_MIN;
	double Umin = DBL_MAX;
	std::vector<Vector2D> X_n;

	for (auto b : nghbr)
	{
		double xi = m.cell[b].centroid().x;
		double yi = m.cell[b].centroid().y;
		X_n.push_back(Vector2D(xi, yi));

		double Ui;
		if (decision == 1)
		{
			Ui = f[b].rho;
		}
		else if (decision == 2)
		{
			Ui = f[b].rhoU.x;
		}
		else if (decision == 3)
		{
			Ui = f[b].rhoU.y;
		}
		else if (decision == 4)
		{
			Ui = f[b].epsilon();
		}

		if (Ui > Umax)
		{
			Umax = Ui;
		}
		else if (Ui < Umin)
		{
			Umin = Ui;
		}
	}

	std::vector<double> psi_n(X_n.size());

#pragma omp parallel for
	for (int i = 0; i<X_n.size(); i++)										// vypocet limiteru
	{
		double grad = dot(GradU(Uc, m, p, f, decision), (X_n[i] - Xc));
		if (grad > 0)
		{

			double temp;
			temp = (Umax - Uc) / grad;
			if (temp > 1)
			{
				psi_n[i] = 1;
			}
			else
			{
				psi_n[i] = temp;
			}
		}
		else if (grad < 0)
		{

			double temp;
			temp = (Umin - Uc) / grad;
			if (temp > 1)
			{
				psi_n[i] = 1;
			}
			else
			{
				psi_n[i] = temp;
			}
		}
		else
		{
			 psi_n[i] = 1;
		}
		
	}

	auto psi_min =  std::min_element(psi_n.begin(),psi_n.end());
	return *psi_min;
	//double psi_min = 0.5;
	//return psi_min;
}

Compressible Method2Order(Field<Compressible> const &f, Compressible Wl, Compressible Wr, Vector2D ne, Mesh const &m, int left, int right, Point edgeCenter)
{																	// Funkce pro vypocet toku velicin metodou 2. radu presnosti

	double g = 0.1;
	
	double xcl = m.cell[left].centroid().x;			// nacteni souradnic
	double ycl = m.cell[left].centroid().y;

	double xcr = m.cell[right].centroid().x;
	double ycr = m.cell[right].centroid().y;

	double xe = edgeCenter.x;
	double ye = edgeCenter.y;

	Vector2D Xcl(xcl, ycl);
	Vector2D Xcr(xcr, ycr);
	Vector2D Xe(xe, ye);

	Vector2D Xl = Xe - Xcl;
	Vector2D Xr = Xe - Xcr;

	double uxcl = Wl.u().x;			// nacteni velicin z vektoru W
	double uxcr = Wr.u().x;

	double uycl = Wl.u().y;
	double uycr = Wr.u().y;

	double rhocl = Wl.rho;
	double rhocr = Wr.rho;

	double epsiloncl = Wl.epsilon();
	double epsiloncr = Wr.epsilon();

	double uxl = uxcl + Psi(uxcl,m,left,f,2) * dot(GradU(uxcl, m, left, f, 2), Xl);				// linearni rekonstrukce velicin ze stredu bunek na stred hrany
	double uxr = uxcr + Psi(uxcr, m, right, f, 2) * dot(GradU(uxcr, m, right, f, 2), Xr);

	double uyl = uycl + Psi(uycl, m, left, f, 3) * dot(GradU(uycl, m, left, f, 3), Xl);
	double uyr = uycr + Psi(uycr, m, right, f, 3) * dot(GradU(uycr, m, right, f, 3), Xr);

	Vector2D ul(uxl, uyl);
	Vector2D ur(uxr, uyr);

	double rhol = rhocl + Psi(rhocl, m, left, f, 1) * dot(GradU(rhocl, m, left, f, 1), Xl);
	double rhor = rhocr + Psi(rhocr, m, right, f, 1) * dot(GradU(rhocr, m, right, f, 1), Xr);

	double epsilonl = epsiloncl + Psi(epsiloncl, m, left, f, 4) * dot(GradU(epsiloncl, m, left, f, 4), Xl);
	double epsilonr = epsiloncr + Psi(epsiloncr, m, right, f, 4) * dot(GradU(epsiloncr, m, right, f, 4), Xr);

	Vector2D ue = 0.5 * (ul + ur);

	double pl = (Compressible::kappa - 1) * rhol * epsilonl;
	double pr = (Compressible::kappa - 1) * rhor * epsilonr;

	double el = rhol * (epsilonl + 0.5 * dot(ul, ul));
	double er = rhor * (epsilonr + 0.5 * dot(ur, ur));

	double pe = 0.5 * (pl + pr);

	if (dot(ue, ne) < 0)								// spetne nacteni velicin do vektoru W
	{
		Compressible F(rhor, rhor * ur, er);
	}
	else
	{
		Compressible F(rhol, rhol * ul, el);
	}

	Compressible F = F * dot(ue, ne);

	F.rhoU = F.rhoU + pe * ne;
	F.e = F.e + pe * dot(ue, ne);

	return F;
}

Compressible fluxNS(Compressible Wl, Compressible Wr, Vector2D ne, Point rl, Point rr)
{

	double mu = 5.0e-5;
	Compressible D;
	Vector2D tau;
	Vector2D taux;
	Vector2D tauy;
	double tauU;

	Vector2D ul = Wl.rhoU / Wl.rho;
	Vector2D ur = Wr.rhoU / Wr.rho;
	Vector2D ue = 0.5 * (ul + ur);

	double rd = pow((rr.x - rl.x), 2) + pow((rr.y - rl.y), 2);
	double a = (double)2 / 3;
	double pe = 0.5 * (Wl.p() + Wr.p());

	double uxx = (ur.x - ul.x) * (rr.x - rl.x) / rd; // dux/dx
	double uxy = (ur.x - ul.x) * (rr.y - rl.y) / rd; // dux/dy
	double uyy = (ur.y - ul.y) * (rr.y - rl.y) / rd; // duy/dy
	double uyx = (ur.y - ul.y) * (rr.x - rl.x) / rd; // duy/dx

	tau.x = mu * ((2 * uxx - a * (uxx + uyy)) * ne.x + (uxy + uyx) * ne.y);
	tau.y = mu * ((uxy + uyx) * ne.x + (2 * uyy - a * (uxx + uyy)) * ne.y);
	tauU = mu * ((2 * uxx - a * (uxx + uyy)) * ue.x * ne.x + (uxy + uyx) * ue.y * ne.x + (uxy + uyx) * ue.x * ne.y + (2 * uyy - a * (uxx + uyy)) * ue.y * ne.y);

	D.rhoU = tau;
	D.e = tauU;

	return D;
}

Compressible fluxHLL(Compressible Wl, Compressible Wr, Vector2D ne)
{
	Vector2D nu = ne / ne.norm();
	double ul = dot(Wl.u(), ne) / ne.norm();
	double ur = dot(Wr.u(), ne) / ne.norm();

	double rhols = std::sqrt(Wl.rho);
	double rhors = std::sqrt(Wr.rho);
	double ub = (rhols * ul + rhors * ur) / (rhols + rhors);
	double Hl = (Wl.e + Wl.p()) / Wl.rho;
	double Hr = (Wr.e + Wr.p()) / Wr.rho;
	double Hb = (rhols * Hl + rhors * Hr) / (rhols + rhors);
	double ab = std::sqrt((Compressible::kappa - 1.0) * (Hb - 0.5 * ub * ub));
	double Sl = ub - ab;
	double Sr = ub + ab;

	Vector2D vl = Wl.u();
	Vector2D vr = Wr.u();
	Vector2D ve = 0.5 * (vl + vr);
	double pe = 0.5 * (Wl.p() + Wr.p());

	Compressible F;
	if (Sl >= 0.0)
	{
		F = Wl * dot(vl, ne) + Compressible(0.0, Wl.p() * ne, Wl.p() * dot(vl, ne));
		// std::cout << "Fl = " << F.rho << " " << F.rhoU.norm() << " " << F.e << "\n";
	}
	else if (Sr <= 0.0)
	{
		F = Wr * dot(vr, ne) + Compressible(0.0, Wr.p() * ne, Wr.p() * dot(vr, ne));
		// std::cout << "Fr = " << F.rho << " " << F.rhoU.norm() << " " << F.e << "\n";
	}
	else
	{
		Compressible Fl = Wl * dot(vl, nu) + Compressible(0.0, Wl.p() * nu, Wl.p() * dot(vl, nu));
		Compressible Fr = Wr * dot(vr, nu) + Compressible(0.0, Wr.p() * nu, Wr.p() * dot(vr, nu));
		F = ne.norm() * (Sr * Fl - Sl * Fr + Sl * Sr * (Wr - Wl)) / (Sr - Sl);
		/*if (Fl.rhoU.x != 0.0) {
			std::cout << "Sr = " << Sr << ", Sl = " << Sl << "\n";
			std::cout << "Fl = " << Fl.rho << " " << Fl.rhoU.x << " " << Fl.rhoU.y << " " << Fl.e << "\n";
			std::cout << "Fr = " << Fr.rho << " " << Fr.rhoU.x << " " << Fr.rhoU.y << " " << Fr.e << "\n";
			std::cout << "Fhll = " << F.rho << " " << F.rhoU.x << " " << F.rhoU.y << " " << F.e << "\n";
		}*/
	}
	return F;
}

double timestep(Mesh const &m, Field<Compressible> const &W)
{
	const double cfl = 0.6; //0.3
	double dt = 1e12;
	double dt_local_min;

#pragma omp parallel private(dt_local_min) shared(dt)
	{
		dt_local_min = 1e12;

#pragma omp for
		for (int i = 0; i < m.nc; ++i)
		{
			Polygon const &p = m.cell[i];
			Vector2D u = W[i].u(); // Fluid velocity
			double c = W[i].c();   // Sound speed
			double lambda = 0.0;

			for (int j = 0; j < p.node_id.size(); ++j)
			{
				int j2;
				if (j == p.node_id.size() - 1)
					j2 = 0;
				else
					j2 = j + 1;
				Point const &n1 = m.node[p.node_id[j]];
				Point const &n2 = m.node[p.node_id[j2]];
				Vector2D e = Vector2D(n1, n2);
				lambda += std::fabs(dot(u, e.normal())) + c * e.norm();
			}
			double dt_i = cfl * p.area() / lambda;
			if (dt_i < dt_local_min)
				dt_local_min = dt_i;
		}

#pragma omp critical
		{
			if (dt_local_min < dt)
				dt = dt_local_min;
		}
	}

	return dt;
}

void FVMstep(Mesh const &m, Field<Compressible> &W, double dt)
{

	Field<Compressible> res(m);

	double g = Compressible::g;

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < m.edge.size(); ++i)
	{
		auto const &e = m.edge[i];
		int l = e.left();  // Index of the cell on the left
		int r = e.right(); // Index of the cell on the right
		// Compressible F = fluxUpwind(W[l],W[r],e.normal());
		Compressible F = Method2Order(W, W[l], W[r], e.normal(), m, l, r, e.center());
		// Compressible F = fluxHLL(W[l], W[r], e.normal());

		//		if (!e.boundary) {
		//		F = F - fluxNS(W[l],W[r],e.normal(),m.cell[l].centroid(),m.cell[r].centroid());
		//		}
#pragma omp critical
		{
			res[l] = res[l] + F;
			if (!e.boundary)
			{
				res[r] = res[r] - F;
			}
		}
	}

#pragma omp parallel for
	for (int j = 0; j < m.nc; ++j)
	{
		//		double const rho = W[j].rho;
		//		Compressible Fg = -Compressible::g*Compressible(0.0,0.0,W[j].rho,W[j].rhoU.y);
		W[j] = W[j] - (dt / m.cell[j].area()) * res[j]; // + dt*Fg;
	}
}
