#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <oneapi/tbb/global_control.h>

#include "fsai.h"
#include "gradiente.h"
#include "matriz.h"

using namespace std;

struct resultado {
    string matriz;
    int n;
    int nnzA;
    int nnzG;
    double construccion;
    double resolucion;
    int iteraciones;
    double residuo;
    bool convergio;
};

bool leer_matriz(const string& archivo, Matriz& A, vector<double>& b, int& n, int& nnz) {
    ifstream entrada(archivo);

    if(!entrada) {
        cerr << "No se pudo abrir " << archivo << "\n";
        return false;
    }

    entrada >> n;
    entrada >> nnz;

    vector<double> valores(nnz);
    for(int i = 0; i < nnz; ++i) {
        entrada >> valores[i];
    }

    vector<int> columnas(nnz);
    for(int i = 0; i < nnz; ++i) {
        entrada >> columnas[i];
    }

    vector<int> inicioFilas(n + 1);
    for(int i = 0; i <= n; ++i) {
        entrada >> inicioFilas[i];
    }

    b.resize(n);
    for(int i = 0; i < n; ++i) {
        entrada >> b[i];
    }

    if(!entrada) {
        cerr << "Error al leer " << archivo << "\n";
        return false;
    }

    A = Matriz(valores, columnas, inicioFilas);

    return true;
}

bool probar_fsai(const string& nombre, double tol, int maxIter, resultado& res) {
    string archivo = "entradas/" + nombre + ".txt";

    cerr << "\n";
    cerr << "============================================================\n";
    cerr << "Matriz: " << nombre << "\n";
    cerr << "============================================================\n";

    Matriz A;
    vector<double> b;
    int n;
    int nnzA;

    cerr << "Leyendo matriz..." << flush;

    if(!leer_matriz(archivo, A, b, n, nnzA)) {
        cerr << " error\n";
        return false;
    }

    cerr << " terminada\n";
    cerr << "n:      " << n << "\n";
    cerr << "nnz(A): " << nnzA << "\n";

    int q = 1;

    fsai prec(A, q);
    gradiente metodoGC(A, b);

    cerr << "Construyendo FSAI q = " << q << "..." << flush;

    auto inicioConstruccion = chrono::steady_clock::now();

    prec.construir();

    auto finConstruccion = chrono::steady_clock::now();

    double construccion =
        chrono::duration<double>(finConstruccion - inicioConstruccion).count();

    int nnzG = prec.nnz();

    cerr << " terminada\n";
    cerr << "nnz(G): " << nnzG << "\n";
    cerr << "Tiempo de construccion: "
         << fixed << setprecision(6)
         << construccion << " s\n";

    cerr << "Resolviendo..." << flush;

    metodoGC.gcp(tol, maxIter, prec);

    const auto& [resolucion, iteraciones, residuo, convergio] =
        metodoGC.getEstado();

    cerr << " terminada\n";
    cerr << "Tiempo de resolucion: "
         << fixed << setprecision(6)
         << resolucion << " s\n";
    cerr << "Iteraciones: " << iteraciones << "\n";
    cerr << "Residuo: "
         << scientific << setprecision(6)
         << residuo << "\n";
    cerr << "Convergio: "
         << (convergio ? "Si" : "No") << "\n";

    res = {
        nombre,
        n,
        nnzA,
        nnzG,
        construccion,
        resolucion,
        iteraciones,
        residuo,
        convergio
    };

    return true;
}

void guardar_resultado(ofstream& salida, const resultado& r) {
    salida << r.matriz << ","
           << r.n << ","
           << r.nnzA << ","
           << r.nnzG << ","
           << fixed << setprecision(6)
           << r.construccion << ","
           << r.resolucion << ","
           << r.construccion + r.resolucion << ","
           << r.iteraciones << ","
           << scientific << setprecision(6)
           << r.residuo << ","
           << (r.convergio ? "Si" : "No")
           << "\n";

    salida.flush();
}

int main() {
    unsigned int hilos = thread::hardware_concurrency();

    if(hilos == 0) {
        hilos = 1;
    }

    oneapi::tbb::global_control limite(
        oneapi::tbb::global_control::max_allowed_parallelism,
        hilos
    );

    const double tol = 1e-9;
    const int maxIter = 1000000;

    vector<string> matrices = {
        "bcsstk17",
        "bcsstk18",
        "bcsstk25",
        "bcsstk36",
        "vanbody",
        "crankseg_1",
        "oilpan",
        "apache1",
        "thermal1",
        "s3dkt3m2",
        "G2_circuit",
        "hood",
        "parabolic_fem",
        "Fault_639",
        "apache2",
        "tmt_sym",
        "ecology2",
        "thermal2",
        "StocF-1465",
        "G3_circuit"
    };

    ofstream salida("salidas/prueba_fsai.csv");

    if(!salida) {
        cerr << "No se pudo crear salidas/prueba_fsai.csv\n";
        return 1;
    }

    salida << "matriz,n,nnzA,nnzG,construccion,resolucion,total,"
           << "iteraciones,residuo,convergio\n";
    salida.flush();

    cerr << "\n";
    cerr << "Prueba general FSAI\n";
    cerr << "Matrices:    " << matrices.size() << "\n";
    cerr << "q:           1\n";
    cerr << "Tolerancia:  " << scientific << tol << "\n";
    cerr << "MaxIter:     " << maxIter << "\n";
    cerr << "Hilos:       " << hilos << "\n";

    int completadas = 0;
    int convergieron = 0;

    for(int i = 0; i < matrices.size(); ++i) {
        cerr << "\n";
        cerr << "Prueba " << i + 1 << "/" << matrices.size() << "\n";

        resultado r;

        try {
            if(probar_fsai(matrices[i], tol, maxIter, r)) {
                guardar_resultado(salida, r);
                ++completadas;

                if(r.convergio) {
                    ++convergieron;
                }
            }
        } catch(const exception& e) {
            cerr << "\nError en " << matrices[i] << ": "
                 << e.what() << "\n";
        }
    }

    salida.close();

    cerr << "\n";
    cerr << "============================================================\n";
    cerr << "Prueba terminada\n";
    cerr << "Matrices completadas: " << completadas
         << "/" << matrices.size() << "\n";
    cerr << "Matrices que convergieron: " << convergieron
         << "/" << matrices.size() << "\n";
    cerr << "Resultados: salidas/prueba_fsai.csv\n";
    cerr << "============================================================\n";

    return 0;
}