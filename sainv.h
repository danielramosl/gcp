#pragma once
#include "matriz.h"
#include "precond.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include "vectores.h"
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

class sainv : public precond {
    private:
    Matriz Z;
    Matriz Zt;
    vector<double> Dinv;
    mutable vector<double> y;
    double tau;

    void Ax_disp(const vector<int>& idX, const vector<double>& valX, vector<double>& res, vector<int>& idRes, vector<int>& marca, int actual) const {
        const vector<double>& valA = A.getVal();
        const vector<int>& idColA = A.getIdCol();
        const vector<int>& iniFilA = A.getIniFil();

        idRes.clear();

        for(int i = 0; i < idX.size(); ++i) {
            int fila = idX[i];
            double valor = valX[i];

            for(int j = iniFilA[fila]; j < iniFilA[fila + 1]; ++j) {
                int pos = idColA[j];

                if(marca[pos] != actual) {
                    marca[pos] = actual;
                    res[pos] = 0.0;
                    idRes.push_back(pos);
                }

                res[pos] += valor * valA[j];
            }
        }
    }

    void agregar(vector<int>& lista, int columna) const {
        if(find(lista.begin(), lista.end(), columna) == lista.end()) {
            lista.push_back(columna);
        }
    }

    void quitar(vector<int>& lista, int columna) const {
        auto it = find(lista.begin(), lista.end(), columna);

        if(it != lista.end()) {
            lista.erase(it);
        }
    }

    void actualizar(vector<int>& idJ, vector<double>& valJ, const vector<int>& idI, const vector<double>& valI, double alpha, int diagonal, vector<vector<int>>& porIndice, vector<int>& nuevoId, vector<double>& nuevoVal) const {
        nuevoId.clear();
        nuevoVal.clear();

        nuevoId.reserve(idJ.size() + idI.size());
        nuevoVal.reserve(valJ.size() + valI.size());

        int a = 0;
        int b = 0;

        while(a < idJ.size() || b < idI.size()) {
            int id;
            double valor;
            bool estaba;

            if(b == idI.size() || (a < idJ.size() && idJ[a] < idI[b])) {
                id = idJ[a];
                valor = valJ[a];
                estaba = true;
                ++a;
            } else if(a == idJ.size() || idI[b] < idJ[a]) {
                id = idI[b];
                valor = -alpha * valI[b];
                estaba = false;
                ++b;
            } else {
                id = idJ[a];
                valor = valJ[a] - alpha * valI[b];
                estaba = true;
                ++a;
                ++b;
            }

            if(id == diagonal || abs(valor) >= tau) {
                nuevoId.push_back(id);
                nuevoVal.push_back(valor);

                if(!estaba) {
                    agregar(porIndice[id], diagonal);
                }
            } else {
                if(estaba) {
                    quitar(porIndice[id], diagonal);
                }
            }
        }

        idJ.swap(nuevoId);
        valJ.swap(nuevoVal);
    }

    public:
    sainv(const Matriz& matriz, double tolerancia) : precond(matriz) {
        tau = tolerancia;
    }

    void construir() override {
        int n = A.getTam();

        vector<vector<int>> idTemp(n);
        vector<vector<double>> valTemp(n);
        vector<vector<int>> porIndice(n);

        vector<double> v(n);
        vector<int> idV;
        vector<int> marcaV(n, -1);

        vector<double> p(n);
        vector<int> candidatos;
        vector<int> marcaP(n, -1);

        vector<int> nuevoId;
        vector<double> nuevoVal;

        Dinv.resize(n);
        y.resize(n);

        for(int i = 0; i < n; ++i) {
            idTemp[i].push_back(i);
            valTemp[i].push_back(1.0);
            porIndice[i].push_back(i);
        }

        for(int i = 0; i < n; ++i) {
            Ax_disp(idTemp[i], valTemp[i], v, idV, marcaV, i);

            double d = 0.0;

            for(int k = 0; k < idTemp[i].size(); ++k) {
                int pos = idTemp[i][k];

                if(marcaV[pos] == i) {
                    d += valTemp[i][k] * v[pos];
                }
            }

            Dinv[i] = 1.0 / d;

            candidatos.clear();

            for(int k = 0; k < idV.size(); ++k) {
                int pos = idV[k];

                if(v[pos] == 0.0) {
                    continue;
                }

                for(int h = 0; h < porIndice[pos].size(); ++h) {
                    int j = porIndice[pos][h];

                    if(j <= i) {
                        continue;
                    }

                    auto it = lower_bound(idTemp[j].begin(), idTemp[j].end(), pos);
                    int indice = it - idTemp[j].begin();

                    if(marcaP[j] != i) {
                        marcaP[j] = i;
                        p[j] = 0.0;
                        candidatos.push_back(j);
                    }

                    p[j] += valTemp[j][indice] * v[pos];
                }
            }

            for(int k = 0; k < candidatos.size(); ++k) {
                int j = candidatos[k];

                if(p[j] == 0.0) {
                    continue;
                }

                double alpha = p[j] * Dinv[i];

                actualizar(idTemp[j], valTemp[j], idTemp[i], valTemp[i], alpha, j, porIndice, nuevoId, nuevoVal);
            }
        }

        int nnzZ = 0;

        for(int i = 0; i < n; ++i) {
            nnzZ += idTemp[i].size();
        }

        vector<double> valZt;
        vector<int> idColZt;
        vector<int> iniFilZt(n + 1);

        valZt.reserve(nnzZ);
        idColZt.reserve(nnzZ);

        iniFilZt[0] = 0;

        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < idTemp[i].size(); ++j) {
                idColZt.push_back(idTemp[i][j]);
                valZt.push_back(valTemp[i][j]);
            }

            iniFilZt[i + 1] = valZt.size();
        }

        Zt = Matriz(valZt, idColZt, iniFilZt);
        Z = Zt.transponer();
    }

    void aplicar(const vector<double>& x, vector<double>& res) const override {
        int n = Dinv.size();

        if(res.size() != n) {
            res.resize(n);
        }

        Zt.Ax(x, y);

        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<int>(0, n, 8192),
            [&](const oneapi::tbb::blocked_range<int>& rango) {
                for(int i = rango.begin(); i < rango.end(); ++i) {
                    y[i] *= Dinv[i];
                }
            }
        );

        Z.Ax(y, res);
    }

    int nnz() const {
        return Z.nnz();
    }
};