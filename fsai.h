#pragma once
#include <algorithm>
#include <cmath>
#include "matriz.h"
#include "precond.h"
#include <vector>

class fsai : public precond {
    private:
    Matriz G;
    Matriz Gt;
    mutable vector<double> y;
    int q;

    vector<vector<int>> patron_estatico() const {
        int n = A.getTam();
        const vector<int>& idCol = A.getIdCol();
        const vector<int>& iniFil = A.getIniFil();
        vector<vector<int>> patron(n);
        for(int i = 0; i < n; ++i) {
            patron[i].push_back(i);
        }
        vector<int> marca(n, -1);
        int actual = 0;
        for(int nivel = 0; nivel < q; ++nivel) {
            vector<vector<int>> nuevoPatron(n);
            for(int i = 0; i < n; ++i) {
                for(int j = 0; j < patron[i].size(); ++j) {
                    int fila = patron[i][j];
                    for(int k = iniFil[fila]; k < iniFil[fila + 1]; ++k) {
                        int columna = idCol[k];
                        if(columna <= i && marca[columna] != actual) {
                            marca[columna] = actual;
                            nuevoPatron[i].push_back(columna);
                        }
                    }
                }
                ++actual;
            }
            patron.swap(nuevoPatron);
        }
        for(int i = 0; i < n; ++i) {
            sort(patron[i].begin(), patron[i].end());
        }
        return patron;
    }

    void extraer_submatriz(const vector<int>& indices, vector<vector<double>>& sub, vector<int>& posicion, vector<int>& marca, int actual) const {
        const vector<double>& val = A.getVal();
        const vector<int>& idCol = A.getIdCol();
        const vector<int>& iniFil = A.getIniFil();
        int m = indices.size();
        sub.assign(m, vector<double>(m, 0.0));
        for(int j = 0; j < m; ++j) {
            int indice = indices[j];
            marca[indice] = actual;
            posicion[indice] = j;
        }
        for(int i = 0; i < m; ++i) {
            int fila = indices[i];
            for(int j = iniFil[fila]; j < iniFil[fila + 1]; ++j) {
                int columna = idCol[j];
                if(marca[columna] == actual) {
                    int colLocal = posicion[columna];
                    sub[i][colLocal] = val[j];
                }
            }
        }
    }

    void resolver_local(vector<vector<double>>& sub, vector<double>& g) const {
        int m = sub.size();
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j <= i; ++j) {
                double suma = sub[i][j];
                for(int k = 0; k < j; ++k) {
                    suma -= sub[i][k] * sub[j][k];
                }
                if(i == j) {
                    sub[i][j] = sqrt(suma);
                } else {
                    sub[i][j] = suma / sub[j][j];
                }
            }
        }
        g.resize(m);
        for(int i = 0; i < m; ++i) {
            double suma = (i == m - 1) ? 1.0 : 0.0;
            for(int j = 0; j < i; ++j) {
                suma -= sub[i][j] * g[j];
            }
            g[i] = suma / sub[i][i];
        }
        for(int i = m - 1; i >= 0; --i) {
            double suma = g[i];
            for(int j = i + 1; j < m; ++j) {
                suma -= sub[j][i] * g[j];
            }
            g[i] = suma / sub[i][i];
        }
    }

    public:
    fsai(const Matriz& matriz, int nivel) : precond(matriz) {
        q = nivel;
    }
    
    void construir() override {
        int n = A.getTam();
        vector<vector<int>> patron = patron_estatico();
        int nnzG = 0;
        for(int i = 0; i < n; ++i) {
            nnzG += patron[i].size();
        }
        vector<double> valG;
        vector<int> idColG;
        vector<int> iniFilG(n + 1);
        valG.reserve(nnzG);
        idColG.reserve(nnzG);
        vector<int> posicion(n);
        vector<int> marca(n, -1);
        vector<vector<double>> sub;
        vector<double> g;
        iniFilG[0] = 0;
        for(int i = 0; i < n; ++i) {
            extraer_submatriz(patron[i], sub, posicion, marca, i);
            resolver_local(sub, g);
            double escala = sqrt(g[g.size() - 1]);
            for(int j = 0; j < g.size(); ++j) {
                g[j] /= escala;
            }
            for(int j = 0; j < patron[i].size(); ++j) {
                idColG.push_back(patron[i][j]);
                valG.push_back(g[j]);
            }
            iniFilG[i + 1] = valG.size();
        }
        G = Matriz(valG, idColG, iniFilG);
        Gt = G.transponer();
        y.resize(n);
    }

    void aplicar(const vector<double>& x, vector<double>& res) const override {
        int n = G.getTam();
        if(res.size() != n) {
            res.resize(n);
        }
        G.Ax(x, y);
        Gt.Ax(y, res);
    }

    int nnz() const {
        return G.nnz();
    }
};