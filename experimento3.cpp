#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <oneapi/tbb/global_control.h>

#include "aib.h"
#include "fsai.h"
#include "gradiente.h"
#include "jacobi.h"
#include "matriz.h"
#include "sainv.h"

using namespace std;

struct resultado {
    string matriz;
    int n;
    int nnzA;
    string metodo;
    string parametros;
    int nnzFactor;
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

template <class P>
resultado evaluar(const string& nombreMatriz, int n, int nnzA, const string& nombreMetodo, const string& parametros, P& prec, gradiente& metodoGC, double tol, int maxIter) {
    cerr << "Construyendo " << nombreMetodo << "..." << flush;

    auto inicio = chrono::steady_clock::now();

    prec.construir();

    auto fin = chrono::steady_clock::now();

    double construccion = chrono::duration<double>(fin - inicio).count();
    int nnzFactor = prec.nnz();

    cerr << " terminada\n";
    cerr << "nnz factor: " << nnzFactor << "\n";
    cerr << "Construccion: "
         << fixed << setprecision(6)
         << construccion << " s\n";

    cerr << "Resolviendo..." << flush;

    metodoGC.gcp(tol, maxIter, prec);

    const auto& [resolucion, iteraciones, residuo, convergio] = metodoGC.getEstado();

    cerr << " terminada\n";
    cerr << "Resolucion: "
         << fixed << setprecision(6)
         << resolucion << " s\n";
    cerr << "Iteraciones: " << iteraciones << "\n";
    cerr << "Residuo: "
         << scientific << setprecision(6)
         << residuo << "\n";
    cerr << "Convergio: "
         << (convergio ? "Si" : "No") << "\n";

    return {
        nombreMatriz,
        n,
        nnzA,
        nombreMetodo,
        parametros,
        nnzFactor,
        construccion,
        resolucion,
        iteraciones,
        residuo,
        convergio
    };
}

void guardar_resultado(ofstream& salida, const resultado& r) {
    salida << r.matriz << ","
           << r.n << ","
           << r.nnzA << ","
           << r.metodo << ","
           << r.parametros << ","
           << r.nnzFactor << ","
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

string parametro_double(const string& nombre, double valor) {
    ostringstream salida;
    salida << nombre << "=" << valor;
    return salida.str();
}

int main() {
    unsigned int hilos = thread::hardware_concurrency();

    if(hilos == 0) {
        hilos = 1;
    }

    oneapi::tbb::global_control limite(oneapi::tbb::global_control::max_allowed_parallelism, hilos);

    const double tol = 1e-10;
    const int maxIter = 1000000;

    const double tauSAINV = 0.2;
    const int qFSAI = 1;
    const double tauAIB = 0.1;
    const int lAIB = 40;

    const string parametrosSAINV = parametro_double("tau", tauSAINV);
    const string parametrosFSAI = "q=" + to_string(qFSAI);
    const string parametrosAIB = parametro_double("tau", tauAIB) + ";l=" + to_string(lAIB);

    vector<string> matrices = {
        "Hook_1498",
        "Geo_1438",
        "Serena",
        "bone010",
        "ldoor"
    };

    ofstream salida("salidas/experimento3.csv");

    if(!salida) {
        cerr << "No se pudo crear salidas/experimento3.csv\n";
        return 1;
    }

    salida << "matriz,n,nnzA,metodo,parametros,nnzFactor,"
           << "construccion,resolucion,total,iteraciones,residuo,convergio\n";
    salida.flush();

    cerr << "\n";
    cerr << "============================================================\n";
    cerr << "EXPERIMENTO 3: MATRICES A GRAN ESCALA\n";
    cerr << "============================================================\n";
    cerr << "Matrices:    " << matrices.size() << "\n";
    cerr << "Metodos:     4\n";
    cerr << "Tolerancia:  " << scientific << tol << "\n";
    cerr << "MaxIter:     " << maxIter << "\n";
    cerr << "Hilos:       " << hilos << "\n";
    cerr << "SAINV tau:   " << tauSAINV << "\n";
    cerr << "FSAI q:      " << qFSAI << "\n";
    cerr << "AIB tau:     " << tauAIB << "\n";
    cerr << "AIB l:       " << lAIB << "\n";

    int pruebas = 0;
    int convergencias = 0;

    for(int m = 0; m < matrices.size(); ++m) {
        const string& nombre = matrices[m];
        string archivo = "entradas/" + nombre + ".txt";

        cerr << "\n";
        cerr << "############################################################\n";
        cerr << "MATRIZ " << m + 1 << "/" << matrices.size()
             << ": " << nombre << "\n";
        cerr << "############################################################\n";

        Matriz A;
        vector<double> b;
        int n;
        int nnzA;

        cerr << "Leyendo matriz..." << flush;

        if(!leer_matriz(archivo, A, b, n, nnzA)) {
            cerr << " error\n";
            continue;
        }

        cerr << " terminada\n";
        cerr << "n:      " << n << "\n";
        cerr << "nnz(A): " << nnzA << "\n";

        gradiente metodoGC(A, b);

        try {
            cerr << "\n";
            cerr << "------------------------------------------------------------\n";
            cerr << "FSAI q = " << qFSAI << "\n";
            cerr << "------------------------------------------------------------\n";

            fsai prec(A, qFSAI);

            resultado r = evaluar(
                nombre,
                n,
                nnzA,
                "FSAI",
                parametrosFSAI,
                prec,
                metodoGC,
                tol,
                maxIter
            );

            guardar_resultado(salida, r);
            ++pruebas;

            if(r.convergio) {
                ++convergencias;
            }
        } catch(const exception& e) {
            cerr << "Error en FSAI: " << e.what() << "\n";
        }

        try {
            cerr << "\n";
            cerr << "------------------------------------------------------------\n";
            cerr << "SAINV tau = " << tauSAINV << "\n";
            cerr << "------------------------------------------------------------\n";

            sainv prec(A, tauSAINV);

            resultado r = evaluar(
                nombre,
                n,
                nnzA,
                "SAINV",
                parametrosSAINV,
                prec,
                metodoGC,
                tol,
                maxIter
            );

            guardar_resultado(salida, r);
            ++pruebas;

            if(r.convergio) {
                ++convergencias;
            }
        } catch(const exception& e) {
            cerr << "Error en SAINV: " << e.what() << "\n";
        }

        try {
            cerr << "\n";
            cerr << "------------------------------------------------------------\n";
            cerr << "AIB tau = " << tauAIB << ", l = " << lAIB << "\n";
            cerr << "------------------------------------------------------------\n";

            aib prec(A, tauAIB, lAIB);

            resultado r = evaluar(
                nombre,
                n,
                nnzA,
                "AIB",
                parametrosAIB,
                prec,
                metodoGC,
                tol,
                maxIter
            );

            guardar_resultado(salida, r);
            ++pruebas;

            if(r.convergio) {
                ++convergencias;
            }
        } catch(const exception& e) {
            cerr << "Error en AIB: " << e.what() << "\n";
        }

        try {
            cerr << "\n";
            cerr << "------------------------------------------------------------\n";
            cerr << "Jacobi\n";
            cerr << "------------------------------------------------------------\n";

            jacobi prec(A);

            resultado r = evaluar(
                nombre,
                n,
                nnzA,
                "Jacobi",
                "-",
                prec,
                metodoGC,
                tol,
                maxIter
            );

            guardar_resultado(salida, r);
            ++pruebas;

            if(r.convergio) {
                ++convergencias;
            }
        } catch(const exception& e) {
            cerr << "Error en Jacobi: " << e.what() << "\n";
        }

        cerr << "\n";
        cerr << "Matriz " << nombre << " terminada.\n";
    }

    salida.close();

    cerr << "\n";
    cerr << "============================================================\n";
    cerr << "EXPERIMENTO TERMINADO\n";
    cerr << "============================================================\n";
    cerr << "Pruebas completadas: " << pruebas << "/"
         << matrices.size() * 4 << "\n";
    cerr << "Convergencias:       " << convergencias << "/"
         << pruebas << "\n";
    cerr << "Resultados:          salidas/experimento3.csv\n";
    cerr << "============================================================\n";

    return 0;
}