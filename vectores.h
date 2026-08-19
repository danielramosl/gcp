#pragma once

#include <cmath>
#include <vector>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>
using namespace std;

vector<double> sumar(const vector<double>& a, const vector<double>& b) {
    vector<double> res(a.size());
    for(int i = 0; i < res.size(); ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}

vector<double> multi_esc(const double c, const vector<int>& x) {
    vector<double> res(x.size());
    for(int i = 0; i < res.size(); ++i) {
        res[i] = c * x[i];
    }
    return res;
}

double norma2(const vector<double>& x) {
    double res = 0;
    for(int i = 0; i < x.size(); ++i) {
        res += x[i] * x[i];
    }
    return sqrt(res);
}

double p_punto(const vector<double>& a, const vector<double>& b) {
    return oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(0, static_cast<int>(a.size()), 8192),
        0.0,
        [&](const oneapi::tbb::blocked_range<int>& rango, double suma) {
            for(int i = rango.begin(); i < rango.end(); ++i) {
                suma += a[i] * b[i];
            }
            return suma;
        },
        [](double a, double b) {
            return a + b;
        }
    );
}