#pragma once
#include "matriz.h"
#include "precond.h"
#include <vector>
#include "vectores.h"
#include <oneapi/tbb/parallel_for.h>

class jacobi : public precond {
    vector<double> invDiag;

    public:
    jacobi(const Matriz& matriz) : precond(matriz) {

    }

    void construir() override {
        const vector<double>& val = A.getVal();
        const vector<int>& idCol = A.getIdCol();
        const vector<int>& iniFil = A.getIniFil();
        int n = iniFil.size() - 1;
        invDiag.resize(n);
        for(int i = 0; i < n; ++i) {
            for(int j = iniFil[i]; j < iniFil[i + 1]; ++j) {
                if(idCol[j] == i) {
                    invDiag[i] = 1.0 / val[j];
                    break;
                }
            }
        }
    }

    void aplicar(const vector<double>& x, vector<double>& res) const override {
        oneapi::tbb::parallel_for(0, static_cast<int>(x.size()), [&](int i) {
            res[i] = invDiag[i] * x[i];
        });
    }

    int nnz() const {
        return invDiag.size();
    }
};