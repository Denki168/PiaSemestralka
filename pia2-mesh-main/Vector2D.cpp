#include "Vector2D.h"

Vector2D operator+(Vector2D const& v, Vector2D const& u){
	Vector2D w(v.x + u.x, v.y + u.y);
	return w;
}

Vector2D operator-(Vector2D const& v, Vector2D const& u){
	Vector2D w(v.x - u.x, v.y - u.y);
	return w;
}

double dot(Vector2D const& v, Vector2D const& u){
    double S;
    S = v.x * u.x + v.y * u.y;
    return S;
}
