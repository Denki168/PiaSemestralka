#ifndef POINT_H
#define POINT_H

#include <vector>
#include <sstream>

class Point {
public:
    Point(double _x, double _y) : x(_x), y(_y) {};
    std::vector<int> cell_ngbr;
    friend std::ostream& operator<<(std::ostream& os, const Point& p);
    double x, y;
};

inline std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << p.x << " " << p.y << " 0.0";
    return os;
};

#endif