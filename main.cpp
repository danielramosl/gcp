#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <oneapi/tbb/global_control.h>

#include "jacobi.h"
#include "gradiente.h"
#include "sainv.h"
#include "fsai.h"
#include "aib.h"

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

template <class P>
resultado evaluar(const string& nombre, P& prec, gradiente& metodoGC, double tol, int maxIter) {
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
    int qFSAI = 1;
    double tauAIB = 0.01;
    int lAIB = 20;

    cerr << "\n";
    cerr << "Dimension:  " << n << "\n";
    cerr << "nnz(A):     " << nnz << "\n";
    cerr << "Hilos:      " << hilos << "\n";
    cerr << "Tolerancia: " << scientific << setprecision(2) << tol << "\n";
    cerr << "Tau SAINV:  " << fixed << setprecision(2) << tauSAINV << "\n";
    cerr << "q FSAI:     " << qFSAI << "\n";
    cerr << "Tau AIB:    " << fixed << setprecision(2) << tauAIB << "\n";
    cerr << "l AIB:      " << lAIB << "\n\n";

    gradiente metodoGC(A, b);

    sainv prec_sainv(A, tauSAINV);
    fsai prec_fsai(A, qFSAI);
    aib prec_aib(A, tauAIB, lAIB);
    jacobi prec_jacobi(A);

    vector<resultado> resultados;

    resultados.push_back(evaluar("GCP SAINV", prec_sainv, metodoGC, tol, maxIter));
    resultados.push_back(evaluar("GCP FSAI", prec_fsai, metodoGC, tol, maxIter));
    resultados.push_back(evaluar("GCP AIB", prec_aib, metodoGC, tol, maxIter));
    resultados.push_back(evaluar("GCP Jacobi", prec_jacobi, metodoGC, tol, maxIter));

    sort(resultados.begin(), resultados.end(), [](const resultado& a, const resultado& b) {
        return a.construccion + a.resolucion < b.construccion + b.resolucion;
    });

    cout << "metodo,nnzM,construccion,resolucion,total,iteraciones,residuo,convergio\n";

    for(int i = 0; i < resultados.size(); ++i) {
        imprimir_csv(resultados[i]);
    }

    return 0;
}