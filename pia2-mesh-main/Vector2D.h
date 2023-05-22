#ifndef VECTOR2D_H
#define VECTOR2D_H

#include <cmath>
#include "Point.h"

class Vector2D {
public:
    Vector2D() : x(0.), y(0.) {};
    Vector2D(double _x, double _y) : x(_x), y(_y) {};
    Vector2D(Point const& u, Point const& v) : x(v.x - u.x), y(v.y - u.y) {};
    friend Vector2D operator+(Vector2D const& v, Vector2D const& u);
    friend Vector2D operator-(Vector2D const& v, Vector2D const& u);
    friend double dot(Vector2D const& v, Vector2D const& u);
    friend Vector2D operator*(double a, Vector2D const& u);
    friend Vector2D operator*(Vector2D const& u, double a);
    friend Vector2D operator/(Vector2D const& u, double a);
    Vector2D normal() const { return Vector2D(y,-x); };
    double norm() const { return std::sqrt(dot(*this,*this)); };
    Vector2D unitNormal() const { return normal()/norm(); };
    double x, y;
};

//multiplication of vector by constant

inline Vector2D operator* (double a, const Vector2D& u){
	Vector2D A;
	
	A.x 	= a * u.x;
	A.y 	= a * u.y;
	
	return A;
}

inline Vector2D operator* (const Vector2D& u, double a){
	Vector2D A;
	
	A.x 	= a * u.x;
	A.y 	= a * u.y;
	
	return A;	
}

//deviding  vector by constant

inline Vector2D operator/ (const Vector2D& u, double a){
	Vector2D A;
	
	A.x 	= u.x / a;
	A.y 	= u.y / a;
	
	return A;	
}

#endif // VECTOR2D_H
