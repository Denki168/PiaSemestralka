#ifndef MESHGMSH_H
#define MESHGMSH_H

#include "Mesh.h"
#include <iostream>
#include <fstream>
#include <string>

class MeshGmsh : public Mesh {
public:
	MeshGmsh(const std::string& name);
private:
	void find_msh_section(std::ifstream& f,const std::string& name);
};

#endif