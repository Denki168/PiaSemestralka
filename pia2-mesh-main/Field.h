#ifndef FIELD_H
#define FIELD_H

#include "Mesh.h"
#include <vector>

template<typename Var>
class Field {
public:
    Field() = delete; // Konstruktor bez parametru je zakazany, musime predat odkaz na objekt se siti
    Field(Mesh const& m) : mesh(m), data(m.cell.size()) {}; // Vytvoreni datoveho pole s velikosti podle poctu bunek site
    Var const& operator[](int i) const { return data[i]; };
    Var& operator[](int i) { return data[i]; }; // Operator [] pujde pouzit primo na objekt tridy Field a datove pole pak nemusi byt primo dostupne
private:
    Mesh const& mesh;
    std::vector<Var> data;
};

#endif // FIELD_H
