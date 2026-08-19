#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "matriz.h"
#include "precond.h"
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

class aib : public precond {
    private:
    Matriz Z;
    Matriz Zt;
    vector<double> Dinv;
    mutable vector<double> y;
    double tau;
    int l;

    void extraer_borde(int k, vector<double>& v, vector<int>& idV, double& alpha) const {
        const vector<double>& val = A.getVal();
        const vector<int>& idCol = A.getIdCol();
        const vector<int>& iniFil = A.getIniFil();

        idV.clear();
        alpha = 0.0;

        for(int j = iniFil[k]; j < iniFil[k + 1]; ++j) {
            int columna = idCol[j];

            if(columna < k) {
                v[columna] = val[j];
                idV.push_back(columna);
            } else if(columna == k) {
                alpha = val[j];
            }
        }
    }

    void calcular_s(const vector<int>& idV, const vector<double>& v, const vector<vector<int>>& idPorFil, const vector<vector<double>>& valPorFil, vector<double>& s, vector<int>& idS, vector<int>& marcaS, int actual) const {
        idS.clear();

        for(int i = 0; i < idV.size(); ++i) {
            int fila = idV[i];
            double valorV = v[fila];

            if(valorV == 0.0) {
                continue;
            }

            for(int j = 0; j < idPorFil[fila].size(); ++j) {
                int columna = idPorFil[fila][j];

                if(marcaS[columna] != actual) {
                    marcaS[columna] = actual;
                    s[columna] = 0.0;
                    idS.push_back(columna);
                }

                s[columna] += valPorFil[fila][j] * valorV;
            }
        }

        for(int i = 0; i < idS.size(); ++i) {
            int pos = idS[i];
            s[pos] *= Dinv[pos];
        }
    }

    void calcular_g(const vector<int>& idS, const vector<double>& s, const vector<vector<int>>& idPorCol, const vector<vector<double>>& valPorCol, vector<double>& g, vector<int>& idG, vector<int>& marcaG, int actual) const {
        idG.clear();

        for(int i = 0; i < idS.size(); ++i) {
            int columna = idS[i];
            double valorS = s[columna];

            if(valorS == 0.0) {
                continue;
            }

            for(int j = 0; j < idPorCol[columna].size(); ++j) {
                int fila = idPorCol[columna][j];

                if(marcaG[fila] != actual) {
                    marcaG[fila] = actual;
                    g[fila] = 0.0;
                    idG.push_back(fila);
                }

                g[fila] -= valPorCol[columna][j] * valorS;
            }
        }
    }

    void truncar(vector<int>& idG, const vector<double>& g, vector<int>& marcaG) const {
        int nuevos = 0;

        for(int i = 0; i < idG.size(); ++i) {
            int pos = idG[i];

            if(g[pos] != 0.0 && abs(g[pos]) >= tau) {
                idG[nuevos] = pos;
                ++nuevos;
            } else {
                marcaG[pos] = -1;
            }
        }

        idG.resize(nuevos);

        if(idG.size() > l) {
            nth_element(idG.begin(), idG.begin() + l, idG.end(), [&g](int a, int b) {
                return abs(g[a]) > abs(g[b]);
            });

            for(int i = l; i < idG.size(); ++i) {
                marcaG[idG[i]] = -1;
            }

            idG.resize(l);
        }

        sort(idG.begin(), idG.end());
    }

    double calcular_delta(double alpha, const vector<int>& idV, const vector<double>& v, const vector<int>& idG, const vector<double>& g, const vector<int>& marcaG, int actual) const {
        const vector<double>& valA = A.getVal();
        const vector<int>& idColA = A.getIdCol();
        const vector<int>& iniFilA = A.getIniFil();

        double vg = 0.0;

        for(int i = 0; i < idV.size(); ++i) {
            int pos = idV[i];

            if(marcaG[pos] == actual) {
                vg += v[pos] * g[pos];
            }
        }

        double gAg = 0.0;

        for(int i = 0; i < idG.size(); ++i) {
            int fila = idG[i];
            double gi = g[fila];

            for(int j = iniFilA[fila]; j < iniFilA[fila + 1]; ++j) {
                int columna = idColA[j];

                if(marcaG[columna] == actual) {
                    gAg += gi * valA[j] * g[columna];
                }
            }
        }

        return alpha + gAg + 2.0 * vg;
    }

    public:
    aib(const Matriz& matriz, double tolerancia, int maxElementos) : precond(matriz) {
        tau = abs(tolerancia);
        l = max(0, maxElementos);
    }

    void construir() override {
        int n = A.getTam();

        vector<vector<int>> idPorCol(n);
        vector<vector<double>> valPorCol(n);

        vector<vector<int>> idPorFil(n);
        vector<vector<double>> valPorFil(n);

        vector<double> v(n);
        vector<int> idV;

        vector<double> s(n);
        vector<int> idS;
        vector<int> marcaS(n, -1);

        vector<double> g(n);
        vector<int> idG;
        vector<int> marcaG(n, -1);

        Dinv.resize(n);
        y.resize(n);

        double alpha;

        extraer_borde(0, v, idV, alpha);

        Dinv[0] = 1.0 / alpha;

        idPorCol[0].push_back(0);
        valPorCol[0].push_back(1.0);

        idPorFil[0].push_back(0);
        valPorFil[0].push_back(1.0);

        int nnzZ = 1;

        for(int k = 1; k < n; ++k) {
            extraer_borde(k, v, idV, alpha);

            calcular_s(idV, v, idPorFil, valPorFil, s, idS, marcaS, k);

            calcular_g(idS, s, idPorCol, valPorCol, g, idG, marcaG, k);

            truncar(idG, g, marcaG);

            double delta = calcular_delta(alpha, idV, v, idG, g, marcaG, k);

            Dinv[k] = 1.0 / delta;

            for(int i = 0; i < idG.size(); ++i) {
                int fila = idG[i];
                double valor = g[fila];

                idPorCol[k].push_back(fila);
                valPorCol[k].push_back(valor);

                idPorFil[fila].push_back(k);
                valPorFil[fila].push_back(valor);
            }

            idPorCol[k].push_back(k);
            valPorCol[k].push_back(1.0);

            idPorFil[k].push_back(k);
            valPorFil[k].push_back(1.0);

            nnzZ += idPorCol[k].size();
        }

        vector<double> valZt;
        vector<int> idColZt;
        vector<int> iniFilZt(n + 1);

        valZt.reserve(nnzZ);
        idColZt.reserve(nnzZ);

        iniFilZt[0] = 0;

        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < idPorCol[i].size(); ++j) {
                idColZt.push_back(idPorCol[i][j]);
                valZt.push_back(valPorCol[i][j]);
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