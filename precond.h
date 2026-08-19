#pragma once
#include "matriz.h"
#include <vector>
#include "vectores.h"

class precond {
protected:
    Matriz A;

    public:
    precond(const Matriz& matriz) : A(matriz) {

    }

    virtual void aplicar(const vector<double>& x, vector<double>& res) const = 0;

    virtual void construir() = 0;

    virtual int nnz() = 0;

    virtual ~precond() {
        
    }
};