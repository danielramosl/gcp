#pragma once

#include "matriz.h"
#include "precond.h"
#include "vectores.h"
#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

class gradiente {
    private:
    Matriz A;
    vector<double> b;
    int n;
    tuple<double, int, double, bool> estado;

    public:
    gradiente(const Matriz& matriz, const vector<double>& vectorB)
        : A(matriz), b(vectorB), n(vectorB.size()), estado(0.0, -1, 0.0, false) {
    }

    vector<double> gc(double tol, int maxIter) {
        auto inicio = chrono::steady_clock::now();
        vector<double> x(n, 0.0);
        vector<double> r = b;
        vector<double> p = r;
        vector<double> Ap(n, 0.0);

        double rr = p_punto(r, r);
        double bb = p_punto(b, b);
        double tol2 = tol * tol * bb;

        int iter = 0;

        while(rr >= tol2 && iter < maxIter) {
            A.Ax(p, Ap);

            double pAp = p_punto(p, Ap);
            double alpha = rr / pAp;

            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<int>(0, n, 8192),
                [&](const oneapi::tbb::blocked_range<int>& rango) {
                    for(int i = rango.begin(); i < rango.end(); ++i) {
                        x[i] += alpha * p[i];
                        r[i] -= alpha * Ap[i];
                    }
                }
            );

            double rrNuevo = p_punto(r, r);
            double beta = rrNuevo / rr;

            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<int>(0, n, 8192),
                [&](const oneapi::tbb::blocked_range<int>& rango) {
                    for(int i = rango.begin(); i < rango.end(); ++i) {
                        p[i] = r[i] + beta * p[i];
                    }
                }
            );

            rr = rrNuevo;
            iter++;
        }

        double residuoFinal = sqrt(rr / bb);
        bool convergio = residuoFinal < tol;

        auto fin = chrono::steady_clock::now();
        double tiempo = chrono::duration<double>(fin - inicio).count();

        estado = make_tuple(tiempo, iter, residuoFinal, convergio);

        return x;
    }

    vector<double> gcp(double tol, int maxIter, const precond& M) {
        auto inicio = chrono::steady_clock::now();
        vector<double> x(n, 0.0);
        vector<double> r = b;
        vector<double> z(n, 0.0);
        vector<double> p(n, 0.0);
        vector<double> Ap(n, 0.0);
        M.aplicar(r, z);
        p = z;
        double rr = p_punto(r, r);
        double rz = p_punto(r, z);
        double bb = p_punto(b, b);
        double tol2 = tol * tol * bb;
        int iter = 0;
        while(rr >= tol2 && iter < maxIter) {
            A.Ax(p, Ap);
            double pAp = p_punto(p, Ap);
            double alpha = rz / pAp;
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<int>(0, n, 8192),
                [&](const oneapi::tbb::blocked_range<int>& rango) {
                    for(int i = rango.begin(); i < rango.end(); ++i) {
                        x[i] += alpha * p[i];
                        r[i] -= alpha * Ap[i];
                    }
                }
            );
            double rrNuevo = p_punto(r, r);
            M.aplicar(r, z);
            double rzNuevo = p_punto(r, z);
            double beta = rzNuevo / rz;
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<int>(0, n, 8192),
                [&](const oneapi::tbb::blocked_range<int>& rango) {
                    for(int i = rango.begin(); i < rango.end(); ++i) {
                        p[i] = z[i] + beta * p[i];
                    }
                }
            );
            rr = rrNuevo;
            rz = rzNuevo;
            iter++;
        }
        double residuoFinal = sqrt(rr / bb);
        bool convergio = residuoFinal < tol;
        auto fin = chrono::steady_clock::now();
        double tiempo = chrono::duration<double>(fin - inicio).count();
        estado = make_tuple(tiempo, iter, residuoFinal, convergio);
        return x;
    }

    const tuple<double, int, double, bool>& getEstado() const {
        return estado;
    }

    string to_string() const {
        const auto& [tiempo, iteraciones, residuoFinal, convergio] = estado;
        if(iteraciones == -1) {
            return "El sistema todavia no ha sido resuelto.";
        }
        ostringstream salida;
        salida << "Tiempo: " << tiempo << " seg\n";
        salida << "Iteraciones: " << iteraciones << "\n";
        salida << "Norma final del residuo: " << residuoFinal << "\n";
        salida << "Convergio: " << (convergio ? "Si" : "No");
        return salida.str();
    }
};