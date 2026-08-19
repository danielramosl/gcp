#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <oneapi/tbb/global_control.h>

#include "gradiente.h"
#include "sainv.h"

using namespace std;

struct resultado {
    string metodo;
    int nnz;
    double construccion;
    double resolucion;
    int iteraciones;
    double residuo;
    bool convergio;
};

resultado evaluar(const string& nombre, sainv& prec, gradiente& metodoGC, double tol, int maxIter) {
    cerr << "Construyendo " << nombre << "..." << flush;

    auto inicio = chrono::steady_clock::now();
    prec.construir();
    auto fin = chrono::steady_clock::now();

    double construccion = chrono::duration<double>(fin - inicio).count();

    cerr << " terminado en " << fixed << setprecision(6) << construccion << " s\n";
    cerr << "Resolviendo con " << nombre << "..." << flush;

    metodoGC.gcp(tol, maxIter, prec);

    auto [resolucion, iteraciones, residuo, convergio] = metodoGC.getEstado();

    cerr << " terminado\n";

    return {nombre, prec.nnz(), construccion, resolucion, iteraciones, residuo, convergio};
}

void imprimir_csv(const resultado& r) {
    cout << r.metodo << ","
         << r.nnz << ","
         << fixed << setprecision(6)
         << r.construccion << ","
         << r.resolucion << ","
         << r.construccion + r.resolucion << ","
         << r.iteraciones << ","
         << scientific << setprecision(6)
         << r.residuo << ","
         << (r.convergio ? "Si" : "No")
         << "\n";
}

int main() {
    unsigned int hilos = thread::hardware_concurrency();

    if(hilos == 0) {
        hilos = 1;
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
    double tauSAINV = 0.1;

    cerr << "\n";
    cerr << "Dimension:  " << n << "\n";
    cerr << "nnz(A):     " << nnz << "\n";
    cerr << "Hilos:      " << hilos << "\n";
    cerr << "Tolerancia: " << scientific << setprecision(2) << tol << "\n";
    cerr << "Tau SAINV:  " << fixed << setprecision(2) << tauSAINV << "\n\n";

    gradiente metodoGC(A, b);
    sainv prec_sainv(A, tauSAINV);

    resultado r = evaluar("GCP SAINV", prec_sainv, metodoGC, tol, maxIter);

    cout << "metodo,nnzM,construccion,resolucion,total,iteraciones,residuo,convergio\n";
    imprimir_csv(r);

    return 0;
}