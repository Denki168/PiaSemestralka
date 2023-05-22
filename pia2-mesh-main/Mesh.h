#ifndef MESH_H
#define MESH_H

#include <vector>
#include <sstream>
#include <set>
#include <cmath>
#include "Point.h"
#include "Vector2D.h"

class Mesh; // Forward declaration required for class Polygon to know
            // about class Mesh.

class Polygon {
public:
    Polygon(std::vector<int> const& _node_id, Mesh & _m) : node_id(_node_id), mesh(_m) {};
    friend std::ostream& operator<<(std::ostream& os, const Polygon& p);
    std::vector<int> node_id;
    Mesh& mesh;
    double area() const;  // trida polygon obsahuje metodu area
    Point centroid() const;
    bool isConvex() const;
    double edgeLength(int i) const; 
};

class Edge{ //stejne jako class ale automaticky vse public
public:
	Edge(int _n1, int _n2, double _hash, Mesh const& _m) : n1(_n1), n2(_n2), hash(_hash), mesh(_m) {};
	Edge(Edge const& e) : n1(e.n1), n2(e.n2), hash(e.hash), mesh(e.mesh), cl(e.cl), cr(e.cr), boundary(e.boundary), location(e.location) {}; // Je nutne specifikovat kopirovaci konstruktor a prirazovaci op.
	Edge& operator=(Edge const& e) { n1 = e.n1; n2 = e.n2; hash = e.hash; cl = e.cl; cr = e.cr; boundary = e.boundary; location = e.location; return *this;}; 
	virtual ~Edge() {};
	double hash;
	int n1, n2;
	//pretizeni operatoru < pro sort objektu, == pro std::unique
	bool operator< (const Edge &other) const { return hash < other.hash; };
    bool operator== (const Edge &other) const { return std::fabs(hash - other.hash) < 1.0e-8; };
	Vector2D normal() const;
	Vector2D unitNormal() const;
	Point center() const;
	// double length();
	int left() const;
	int right() const;
	int cl, cr;
	bool boundary;
	int location;
private:
	Mesh const& mesh;
};

class Mesh {
public:
	Mesh() {};
    Mesh(double xl, double xr, double yl, double yr, int nx, int ny);
    std::vector<Polygon> cell;
    int nc; // Number of internal (non-ghost) cells
    
    std::vector<Point> node;
	int nCellNodes() const; // Total number of all cell corners
	
    std::vector<Edge> edge;
	void initPointCellNeighbors();
    void generateEdges();
    void initLeftRight();
    void addGhostCells();
    std::set<int> boundaryNodes;
    void randomize(double r);
	std::vector<std::vector<int> > edgeNeighbors() const;
};

#endif //MESH_H
