#ifndef COMPRESSIBLE_H
#define COMPRESSIBLE_H

#include "Mesh.h"
#include "Field.h"
#include "Vector2D.h"

class Compressible
{
public:
    Compressible() : rho(0.0), rhoU(0.0, 0.0), e(0.0){};
    Compressible(double _rho, double _rhoux, double _rhouy, double _e) : rho(_rho), rhoU(_rhoux, _rhouy), e(_e){};
    Compressible(double _rho, Vector2D _rhou, double _e) : rho(_rho), rhoU(_rhou), e(_e){};
    constexpr static double kappa = 1.4;
    static double g;
    double rho, e;
    Vector2D rhoU;
    double epsilon() const;
    double p() const;
    double c() const;
    Vector2D u() const;
    double eos_e_from_p(double p_in) const; // Calculate total energy density from the EOS using pressure value
    Compressible &operator-=(Compressible const &a)
    {
        rho -= a.rho;
        rhoU = rhoU - a.rhoU;
        e -= a.e;
        return *this;
    }
};

inline Compressible operator+(Compressible const &a, Compressible const &b)
{
    return Compressible(a.rho + b.rho, a.rhoU.x + b.rhoU.x, a.rhoU.y + b.rhoU.y, a.e + b.e);
}
inline Compressible operator-(Compressible const &a, Compressible const &b)
{
    return Compressible(a.rho - b.rho, a.rhoU.x - b.rhoU.x, a.rhoU.y - b.rhoU.y, a.e - b.e);
}
inline Compressible operator*(double a, Compressible const &b)
{
    return Compressible(a * b.rho, a * b.rhoU.x, a * b.rhoU.y, a * b.e);
}
inline Compressible operator*(Compressible const &b, double a)
{
    return Compressible(a * b.rho, a * b.rhoU.x, a * b.rhoU.y, a * b.e);
}
inline Compressible operator/(Compressible const &b, double a)
{
    return Compressible(b.rho / a, b.rhoU.x / a, b.rhoU.y / a, b.e / a);
}

Vector2D GradU(double Uc, Mesh const &m, int p, Field<Compressible> const &f, int decision);
double Psi(double Uc, Mesh const &m, int p, Field<Compressible> const &f, int decision);
Compressible Method2Order(Field<Compressible> const &f, Compressible Wl, Compressible Wr, Vector2D ne, Mesh const &m, int left, int right, Point edgeCenter);
Compressible fluxUpwind(Compressible Wl, Compressible Wr, Vector2D ne);
Compressible fluxHLL(Compressible Wl, Compressible Wr, Vector2D ne);
double timestep(Mesh const &m, Field<Compressible> const &W);
void FVMstep(Mesh const &m, Field<Compressible> &W, double dt);

#endif // COMPRESSIBLE_H
