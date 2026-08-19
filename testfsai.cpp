#include <iostream>
#include <vector>

#include "fsai.h"

using namespace std;

void imprimir_matriz(const vector<vector<double>>& M) {
    for(int i = 0; i < M.size(); ++i) {
        for(int j = 0; j < M[i].size(); ++j) {
            cout << M[i][j] << " ";
        }
        cout << "\n";
    }
}

int main() {
    vector<double> val = {
        4, -1,
        -1, 4, -1,
        -1, 4, -1,
        -1, 4, -1,
        -1, 4
    };

    vector<int> idCol = {
        0, 1,
        0, 1, 2,
        1, 2, 3,
        2, 3, 4,
        3, 4
    };

    vector<int> iniFil = {
        0, 2, 5, 8, 11, 13
    };

    Matriz A(val, idCol, iniFil);

    fsai prec(A, 2);

    vector<int> indices = {0, 2, 4};

    vector<vector<double>> sub;

    vector<int> posicion(A.getTam());
    vector<int> marca(A.getTam(), -1);

    prec.extraer_submatriz(
        indices,
        sub,
        posicion,
        marca,
        0
    );

    cout << "A[P,P]:\n";
    imprimir_matriz(sub);

    return 0;
}