#include <algorithm>
#include <chrono>
#include <cmath>
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

struct piloto {
    double tiempo;
    int nnz;
};

struct configSAINV {
    double tau;
    piloto p;
};

struct configAIB {
    double tau;
    int l;
    piloto p;
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
    cerr << "Construyendo " << nombreMetodo << " " << parametros << "..." << flush;

    auto inicio = chrono::steady_clock::now();

    prec.construir();

    auto fin = chrono::steady_clock::now();

    double construccion = chrono::duration<double>(fin - inicio).count();
    int nnzFactor = prec.nnz();

    cerr << " terminada\n";
    cerr << "nnz factor:   " << nnzFactor << "\n";
    cerr << "Construccion: " << fixed << setprecision(6) << construccion << " s\n";

    cerr << "Resolviendo..." << flush;

    metodoGC.gcp(tol, maxIter, prec);

    const auto& [resolucion, iteraciones, residuo, convergio] = metodoGC.getEstado();

    cerr << " terminada\n";
    cerr << "Resolucion:   " << fixed << setprecision(6) << resolucion << " s\n";
    cerr << "Total:        " << construccion + resolucion << " s\n";
    cerr << "Iteraciones:  " << iteraciones << "\n";
    cerr << "Residuo:      " << scientific << setprecision(6) << residuo << "\n";
    cerr << "Convergio:    " << (convergio ? "Si" : "No") << "\n";

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

double total(const resultado& r) {
    return r.construccion + r.resolucion;
}

bool es_mejor(const resultado& candidato, const resultado& mejor, bool hayMejor) {
    if(!hayMejor) {
        return true;
    }

    if(candidato.convergio != mejor.convergio) {
        return candidato.convergio;
    }

    return total(candidato) < total(mejor);
}

string numero(double x) {
    ostringstream salida;
    salida << defaultfloat << setprecision(6) << x;
    return salida.str();
}

Matriz submatriz_principal(const Matriz& A, int m) {
    const vector<double>& val = A.getVal();
    const vector<int>& idCol = A.getIdCol();
    const vector<int>& iniFil = A.getIniFil();

    m = min(m, A.getTam());

    vector<double> valSub;
    vector<int> idColSub;
    vector<int> iniFilSub(m + 1);

    iniFilSub[0] = 0;

    for(int i = 0; i < m; ++i) {
        for(int j = iniFil[i]; j < iniFil[i + 1]; ++j) {
            if(idCol[j] < m) {
                valSub.push_back(val[j]);
                idColSub.push_back(idCol[j]);
            }
        }

        iniFilSub[i + 1] = valSub.size();
    }

    return Matriz(valSub, idColSub, iniFilSub);
}

template <class P>
piloto medir_piloto(P& prec) {
    auto inicio = chrono::steady_clock::now();

    prec.construir();

    auto fin = chrono::steady_clock::now();

    return {
        chrono::duration<double>(fin - inicio).count(),
        prec.nnz()
    };
}

double estimar_construccion(double construccionReferencia, const piloto& referencia, const piloto& candidato) {
    double razonTiempo = candidato.tiempo / max(referencia.tiempo, 1e-9);
    double razonNnz = static_cast<double>(candidato.nnz) / max(referencia.nnz, 1);

    return construccionReferencia * max(razonTiempo, razonNnz);
}

vector<int> filas_muestra_fsai(const Matriz& A, int muestras) {
    int n = A.getTam();

    muestras = min(muestras, n);

    const vector<int>& idCol = A.getIdCol();
    const vector<int>& iniFil = A.getIniFil();

    vector<int> tamQ1(n, 0);

    for(int i = 0; i < n; ++i) {
        for(int j = iniFil[i]; j < iniFil[i + 1]; ++j) {
            if(idCol[j] <= i) {
                ++tamQ1[i];
            }
        }
    }

    vector<char> usado(n, 0);
    vector<int> filas;

    int uniformes = min(3 * muestras / 4, n);

    if(uniformes == 1) {
        filas.push_back(0);
        usado[0] = 1;
    } else if(uniformes > 1) {
        for(int s = 0; s < uniformes; ++s) {
            int i = static_cast<long long>(s) * (n - 1) / (uniformes - 1);

            if(!usado[i]) {
                usado[i] = 1;
                filas.push_back(i);
            }
        }
    }

    vector<int> orden(n);

    for(int i = 0; i < n; ++i) {
        orden[i] = i;
    }

    sort(orden.begin(), orden.end(), [&](int a, int b) {
        return tamQ1[a] > tamQ1[b];
    });

    for(int k = 0; k < n && filas.size() < muestras; ++k) {
        int i = orden[k];

        if(!usado[i]) {
            usado[i] = 1;
            filas.push_back(i);
        }
    }

    return filas;
}

int tam_patron_fsai_fila(const Matriz& A, int fila, int q, vector<int>& marca, int& actual, vector<int>& patron, vector<int>& nuevoPatron) {
    const vector<int>& idCol = A.getIdCol();
    const vector<int>& iniFil = A.getIniFil();

    patron.clear();
    patron.push_back(fila);

    for(int nivel = 0; nivel < q; ++nivel) {
        nuevoPatron.clear();
        ++actual;

        for(int j = 0; j < patron.size(); ++j) {
            int f = patron[j];

            for(int k = iniFil[f]; k < iniFil[f + 1]; ++k) {
                int columna = idCol[k];

                if(columna <= fila && marca[columna] != actual) {
                    marca[columna] = actual;
                    nuevoPatron.push_back(columna);
                }
            }
        }

        patron.swap(nuevoPatron);
    }

    return patron.size();
}

long double costo_muestra_fsai(const Matriz& A, int q, const vector<int>& filas) {
    int n = A.getTam();

    vector<int> marca(n, -1);
    vector<int> patron;
    vector<int> nuevoPatron;

    int actual = 0;

    long double costo = 0.0L;

    for(int i = 0; i < filas.size(); ++i) {
        long double m = tam_patron_fsai_fila(A, filas[i], q, marca, actual, patron, nuevoPatron);

        costo += m * m * m;
    }

    return costo;
}

bool ajustar_sainv(const string& nombre, int n, int nnzA, const Matriz& A, gradiente& metodoGC, double tol, int maxIter, double margenPoda, resultado& mejor) {
    vector<double> taus = {
        0.01,
        0.05,
        0.1,
        0.2,
        0.5
    };

    const int tamPiloto = 1500;

    if(n <= 2 * tamPiloto) {
        bool hayMejor = false;

        for(int i = 0; i < taus.size(); ++i) {
            try {
                sainv prec(A, taus[i]);

                resultado r = evaluar(nombre, n, nnzA, "SAINV", "tau=" + numero(taus[i]), prec, metodoGC, tol, maxIter);

                if(es_mejor(r, mejor, hayMejor)) {
                    mejor = r;
                    hayMejor = true;
                }
            } catch(const exception& e) {
                cerr << "Error SAINV tau=" << taus[i] << ": " << e.what() << "\n";
            }
        }

        return hayMejor;
    }

    Matriz APiloto = submatriz_principal(A, tamPiloto);

    vector<configSAINV> configs;

    cerr << "\n";
    cerr << "PILOTO SAINV\n";
    cerr << "Tamano: " << tamPiloto << "\n";

    for(int i = 0; i < taus.size(); ++i) {
        try {
            sainv prec(APiloto, taus[i]);

            piloto p = medir_piloto(prec);

            configs.push_back({
                taus[i],
                p
            });

            cerr << "tau=" << taus[i]
                 << " tiempo=" << fixed << setprecision(6) << p.tiempo
                 << " nnz=" << p.nnz << "\n";
        } catch(const exception& e) {
            cerr << "Error piloto SAINV tau=" << taus[i] << ": " << e.what() << "\n";
        }
    }

    sort(configs.begin(), configs.end(), [](const configSAINV& a, const configSAINV& b) {
        if(a.p.tiempo == b.p.tiempo) {
            return a.p.nnz < b.p.nnz;
        }

        return a.p.tiempo < b.p.tiempo;
    });

    bool hayMejor = false;
    bool hayReferencia = false;

    piloto referenciaPiloto;
    double construccionReferencia = 0.0;

    for(int i = 0; i < configs.size(); ++i) {
        if(hayReferencia && hayMejor) {
            double estimada = estimar_construccion(construccionReferencia, referenciaPiloto, configs[i].p);
            double limite = margenPoda * total(mejor);

            cerr << "\n";
            cerr << "SAINV tau=" << configs[i].tau << "\n";
            cerr << "Construccion estimada: " << fixed << setprecision(6) << estimada << " s\n";
            cerr << "Limite de poda:         " << limite << " s\n";

            if(estimada > limite) {
                cerr << "PODADO\n";
                continue;
            }
        }

        try {
            sainv prec(A, configs[i].tau);

            resultado r = evaluar(nombre, n, nnzA, "SAINV", "tau=" + numero(configs[i].tau), prec, metodoGC, tol, maxIter);

            if(es_mejor(r, mejor, hayMejor)) {
                mejor = r;
                hayMejor = true;
            }

            referenciaPiloto = configs[i].p;
            construccionReferencia = r.construccion;
            hayReferencia = true;
        } catch(const exception& e) {
            cerr << "Error SAINV tau=" << configs[i].tau << ": " << e.what() << "\n";
        }
    }

    return hayMejor;
}

bool ajustar_fsai(const string& nombre, int n, int nnzA, const Matriz& A, gradiente& metodoGC, double tol, int maxIter, double margenPoda, resultado& mejor) {
    const int muestras = 128;

    vector<int> filas = filas_muestra_fsai(A, muestras);

    long double costoAnterior = costo_muestra_fsai(A, 1, filas);

    bool hayMejor = false;

    resultado anterior;

    try {
        fsai prec(A, 1);

        resultado r = evaluar(nombre, n, nnzA, "FSAI", "q=1", prec, metodoGC, tol, maxIter);

        mejor = r;
        anterior = r;
        hayMejor = true;
    } catch(const exception& e) {
        cerr << "Error FSAI q=1: " << e.what() << "\n";
        return false;
    }

    for(int q = 2; q <= 3; ++q) {
        long double costoActual = costo_muestra_fsai(A, q, filas);

        long double razon = costoActual / costoAnterior;

        double estimada = anterior.construccion * static_cast<double>(razon);
        double limite = margenPoda * total(mejor);

        cerr << "\n";
        cerr << "FSAI q=" << q << "\n";
        cerr << "Crecimiento estructural estimado: "
             << fixed << setprecision(3)
             << static_cast<double>(razon) << "x\n";
        cerr << "Construccion estimada:            "
             << setprecision(6)
             << estimada << " s\n";
        cerr << "Limite de poda:                    "
             << limite << " s\n";

        if(!isfinite(estimada) || estimada > limite) {
            cerr << "PODADO\n";

            break;
        }

        try {
            fsai prec(A, q);

            resultado r = evaluar(nombre, n, nnzA, "FSAI", "q=" + to_string(q), prec, metodoGC, tol, maxIter);

            if(es_mejor(r, mejor, hayMejor)) {
                mejor = r;
            }

            anterior = r;
            costoAnterior = costoActual;
        } catch(const exception& e) {
            cerr << "Error FSAI q=" << q << ": " << e.what() << "\n";

            break;
        }
    }

    return hayMejor;
}

bool ajustar_aib(const string& nombre, int n, int nnzA, const Matriz& A, gradiente& metodoGC, double tol, int maxIter, double margenPoda, resultado& mejor) {
    vector<double> taus = {
        0.001,
	0.01,
	0.1
    };

    vector<int> ls = {
        5,
        10,
        20,
        40
    };

    const int tamPiloto = 1500;

    if(n <= 2 * tamPiloto) {
        bool hayMejor = false;

        for(int i = 0; i < taus.size(); ++i) {
            for(int j = 0; j < ls.size(); ++j) {
                try {
                    aib prec(A, taus[i], ls[j]);

                    resultado r = evaluar(nombre, n, nnzA, "AIB", "tau=" + numero(taus[i]) + ";l=" + to_string(ls[j]), prec, metodoGC, tol, maxIter);

                    if(es_mejor(r, mejor, hayMejor)) {
                        mejor = r;
                        hayMejor = true;
                    }
                } catch(const exception& e) {
                    cerr << "Error AIB tau=" << taus[i]
                         << " l=" << ls[j]
                         << ": " << e.what() << "\n";
                }
            }
        }

        return hayMejor;
    }

    Matriz APiloto = submatriz_principal(A, tamPiloto);

    vector<configAIB> configs;

    cerr << "\n";
    cerr << "PILOTO AIB\n";
    cerr << "Tamano: " << tamPiloto << "\n";

    for(int i = 0; i < taus.size(); ++i) {
        for(int j = 0; j < ls.size(); ++j) {
            try {
                aib prec(APiloto, taus[i], ls[j]);

                piloto p = medir_piloto(prec);

                configs.push_back({
                    taus[i],
                    ls[j],
                    p
                });

                cerr << "tau=" << taus[i]
                     << " l=" << ls[j]
                     << " tiempo=" << fixed << setprecision(6) << p.tiempo
                     << " nnz=" << p.nnz << "\n";
            } catch(const exception& e) {
                cerr << "Error piloto AIB tau=" << taus[i]
                     << " l=" << ls[j]
                     << ": " << e.what() << "\n";
            }
        }
    }

    sort(configs.begin(), configs.end(), [](const configAIB& a, const configAIB& b) {
        if(a.p.tiempo == b.p.tiempo) {
            return a.p.nnz < b.p.nnz;
        }

        return a.p.tiempo < b.p.tiempo;
    });

    bool hayMejor = false;
    bool hayReferencia = false;

    piloto referenciaPiloto;
    double construccionReferencia = 0.0;

    for(int i = 0; i < configs.size(); ++i) {
        if(hayReferencia && hayMejor) {
            double estimada = estimar_construccion(construccionReferencia, referenciaPiloto, configs[i].p);
            double limite = margenPoda * total(mejor);

            cerr << "\n";
            cerr << "AIB tau=" << configs[i].tau
                 << " l=" << configs[i].l << "\n";
            cerr << "Construccion estimada: " << fixed << setprecision(6) << estimada << " s\n";
            cerr << "Limite de poda:         " << limite << " s\n";

            if(estimada > limite) {
                cerr << "PODADO\n";
                continue;
            }
        }

        try {
            aib prec(A, configs[i].tau, configs[i].l);

            resultado r = evaluar(nombre, n, nnzA, "AIB", "tau=" + numero(configs[i].tau) + ";l=" + to_string(configs[i].l), prec, metodoGC, tol, maxIter);

            if(es_mejor(r, mejor, hayMejor)) {
                mejor = r;
                hayMejor = true;
            }

            referenciaPiloto = configs[i].p;
            construccionReferencia = r.construccion;
            hayReferencia = true;
        } catch(const exception& e) {
            cerr << "Error AIB tau=" << configs[i].tau
                 << " l=" << configs[i].l
                 << ": " << e.what() << "\n";
        }
    }

    return hayMejor;
}

int main() {
    unsigned int hilos = thread::hardware_concurrency();

    if(hilos == 0) {
        hilos = 1;
    }

    oneapi::tbb::global_control limite(oneapi::tbb::global_control::max_allowed_parallelism, hilos);

    const double tol = 1e-10;
    const int maxIter = 1000000;

    const double margenPoda = 2.0;

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

    ofstream salida("salidas/ajuste_hiperparametros.csv");

    if(!salida) {
        cerr << "No se pudo crear salidas/ajuste_hiperparametros.csv\n";
        return 1;
    }

    salida << "matriz,n,nnzA,metodo,parametros,nnzFactor,"
           << "construccion,resolucion,total,iteraciones,residuo,convergio\n";

    salida.flush();

    cerr << "\n";
    cerr << "============================================================\n";
    cerr << "AJUSTE DE HIPERPARAMETROS\n";
    cerr << "============================================================\n";
    cerr << "Matrices:       " << matrices.size() << "\n";
    cerr << "Tolerancia:     " << scientific << tol << "\n";
    cerr << "MaxIter:        " << maxIter << "\n";
    cerr << "Hilos:          " << hilos << "\n";
    cerr << "Margen de poda: " << fixed << setprecision(2) << margenPoda << "x\n";

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

        if(!leer_matriz(archivo, A, b, n, nnzA)) {
            continue;
        }

        cerr << "n:      " << n << "\n";
        cerr << "nnz(A): " << nnzA << "\n";

        gradiente metodoGC(A, b);

        resultado mejor;

        if(ajustar_sainv(nombre, n, nnzA, A, metodoGC, tol, maxIter, margenPoda, mejor)) {
            guardar_resultado(salida, mejor);

            cerr << "\n";
            cerr << "MEJOR SAINV: "
                 << mejor.parametros
                 << " total="
                 << fixed << setprecision(6)
                 << total(mejor) << " s\n";
        }

        if(ajustar_fsai(nombre, n, nnzA, A, metodoGC, tol, maxIter, margenPoda, mejor)) {
            guardar_resultado(salida, mejor);

            cerr << "\n";
            cerr << "MEJOR FSAI: "
                 << mejor.parametros
                 << " total="
                 << fixed << setprecision(6)
                 << total(mejor) << " s\n";
        }

        if(ajustar_aib(nombre, n, nnzA, A, metodoGC, tol, maxIter, margenPoda, mejor)) {
            guardar_resultado(salida, mejor);

            cerr << "\n";
            cerr << "MEJOR AIB: "
                 << mejor.parametros
                 << " total="
                 << fixed << setprecision(6)
                 << total(mejor) << " s\n";
        }

        cerr << "\n";
        cerr << "Matriz " << nombre << " terminada.\n";
    }

    salida.close();

    cerr << "\n";
    cerr << "============================================================\n";
    cerr << "AJUSTE TERMINADO\n";
    cerr << "============================================================\n";
    cerr << "Resultados: salidas/ajuste_hiperparametros.csv\n";
    cerr << "============================================================\n";

    return 0;
}
