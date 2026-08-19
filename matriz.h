#pragma once
#include <algorithm>
#include <vector>
#include <oneapi/tbb/parallel_for.h>
using namespace std;

class Matriz {
    int tam;
    vector<double> val;
    vector<int> idCol;
    vector<int> iniFil;

    public:
    Matriz(const vector<double>& val, const vector<int>& idCol,
           const vector<int>& iniFil) {
        this->val = val;
        this->idCol = idCol;
        this->iniFil = iniFil;
        tam = iniFil.size() - 1;
    }

    Matriz() {
        tam = 0;
    }

    void Ax(const vector<double>& x, vector<double>& res) const {
        oneapi::tbb::parallel_for(0, tam, [&](int i) {
            double suma = 0.0;
            for(int j = iniFil[i]; j < iniFil[i + 1]; ++j) {
                suma += val[j] * x[idCol[j]];
            }
            res[i] = suma;
        });
    }

    Matriz transponer() const {
        vector<double> valT(val.size());
        vector<int> idColT(val.size());
        vector<int> iniFilT(tam + 1, 0);
        for(int i = 0; i < idCol.size(); ++i) {
            ++iniFilT[idCol[i] + 1];
        }
        for(int i = 0; i < tam; ++i) {
            iniFilT[i + 1] += iniFilT[i];
        }
        vector<int> siguiente = iniFilT;
        for(int i = 0; i < tam; ++i) {
            for(int j = iniFil[i]; j < iniFil[i + 1]; ++j) {
                int fila = idCol[j];
                int pos = siguiente[fila];

                idColT[pos] = i;
                valT[pos] = val[j];

                ++siguiente[fila];
            }
        }
        return Matriz(valT, idColT, iniFilT);
    }

    const vector<double>& getVal() const {
        return val;
    }

    const vector<int>& getIdCol() const {
        return idCol;
    }

    const vector<int>& getIniFil() const {
       return iniFil;
    }

    const int getTam() const {
        return tam;
    }

    int nnz() const {
        return val.size();
    }
};