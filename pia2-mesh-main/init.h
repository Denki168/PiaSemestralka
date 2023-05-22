#ifndef INIT_H
#define INIT_H
#include <memory>
#include "Mesh.h"
#include "Field.h"
#include "Compressible.h"
#include "BC.h"

void initSod(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds);
void initJet(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds);
void initKH(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds);
void initRayTay(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds, int reg);
void initRayTayCos(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds, int reg);
void initVazDesk(Mesh *& m, Field<Compressible> *& W, std::vector<BC<Compressible>*>& boundary_conds);

#endif
