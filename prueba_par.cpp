#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>

#include "jacobi.h"
#include "gradiente.h"
#include "fsai.h"
#include "aib.h"

using namespace std;

struct resultado {
    string metodo;
    int hilos;
    int nnz;
    double construccion;
    double resolucion;
    int iteraciones;
    double residuo;
    bool convergio;
};

void imprimir_fila(const resultado& r) {
    cout << left << setw(22) << r.metodo;

    cout << right
         << setw(8) << r.hilos
         << setw(14) << r.nnz
         << fixed << setprecision(6)
         << setw(16) << r.construccion
         << setw(16) << r.resolucion
         << setw(16) << r.construccion + r.resolucion
         << setw(14) << r.iteraciones;

    cout << scientific << setprecision(6)
         << setw(18) << r.residuo;

    cout << setw(12) << (r.convergio ? "Si" : "No") << "\n";
}

int main(int argc, char* argv[]) {
    int hilos;

    if(argc >= 2) {
        hilos = stoi(argv[1]);
    } else {
        hilos = oneapi::tbb::info::default_concurrency();
    }

    oneapi::tbb::global_control limite(oneapi::tbb::global_control::max_allowed_parallelism, hilos);

    int n;
    int nnz;

    cin >> n;
    cin >> nnz;

    vector<double> valores(nnz);
    for(int i = 0; i < nnz; ++i) {
        cin >> valores[i];
    }

    vector<int> columnas(nnz);
    for(int i = 0; i < nnz; ++i) {
        cin >> columnas[i];
    }

    vector<int> inicioFilas(n + 1);
    for(int i = 0; i < n + 1; ++i) {
        cin >> inicioFilas[i];
    }

    vector<double> b(n);
    for(int i = 0; i < n; ++i) {
        cin >> b[i];
    }

    Matriz A(valores, columnas, inicioFilas);

    double tol = 1e-10;
    int maxIter = 1000000;

    gradiente metodoGC(A, b);
    jacobi pre(A);

    auto iniciojacobi = chrono::steady_clock::now();

    pre.construir();

    auto finjacobi = chrono::steady_clock::now();

    double construccionjacobi = chrono::duration<double>(finjacobi - iniciojacobi).count();

    metodoGC.gcp(tol, maxIter, pre);

    auto [tiempojacobi, iterjacobi, residuojacobi, convergiojacobi] = metodoGC.getEstado();

    resultado res = {
        "GCP jacobi",
        hilos,
        pre.nnz(),
        construccionjacobi,
        tiempojacobi,
        iterjacobi,
        residuojacobi,
        convergiojacobi
    };

    imprimir_fila(res);

    return 0;
}