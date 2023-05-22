#ifndef BC_H_
#define BC_H_
#include "Compressible.h"
#include "Field.h"

template <typename T>
class BC {
public:
	BC(std::vector<int> loc_in) : locations(loc_in) {};
	virtual ~BC() {};
	std::vector<int> locations;
	bool isCorrectLocation(int loc) {
		for (int l : locations) {
			if (l==loc) return true;
		}
		return false;
	};
	virtual void apply(Mesh const& m, Field<Compressible> & W) = 0;
};

class SlipWallBC : public BC<Compressible> {
public:
	using BC::BC; // C++11 directive for inheritance of BC class constructors
	virtual ~SlipWallBC() {};
	virtual void apply(Mesh const& m, Field<Compressible> & W);
};

class ReservoirBC : public BC<Compressible> {
public:
	using BC::BC; // C++11 directive for inheritance of BC class constructors
	virtual ~ReservoirBC() {};
	virtual void apply(Mesh const& m, Field<Compressible> & W);
};

class ConstVelocityBC : public BC<Compressible> {
public:
	ConstVelocityBC(std::vector<int> loc_in, Vector2D velocity_in) : BC(loc_in), velocity(velocity_in) {};
	virtual ~ConstVelocityBC() {};
	virtual void apply(Mesh const& m, Field<Compressible> & W);
private:
    Vector2D velocity;
};

class puBC : public BC<Compressible> {
public:
	using BC::BC; // C++11 directive for inheritance of BC class constructors
	virtual ~puBC() {};
	virtual void apply(Mesh const& m, Field<Compressible> & W);
};

class muBC : public BC<Compressible> {
public:
	using BC::BC; // C++11 directive for inheritance of BC class constructors
	virtual ~muBC() {};
	virtual void apply(Mesh const& m, Field<Compressible> & W);
};

class FreeBC : public BC<Compressible> {
public:
	using BC::BC; // C++11 directive for inheritance of BC class constructors
	virtual ~FreeBC() {};
	virtual void apply(Mesh const& m, Field<Compressible> & W);
};
//na desku dole vpravo
class vazDeskBC : public BC<Compressible> {
public:
	using BC::BC; // C++11 directive for inheritance of BC class constructors
	virtual ~vazDeskBC() {};
	virtual void apply(Mesh const& m, Field<Compressible> & W);
};

#endif
