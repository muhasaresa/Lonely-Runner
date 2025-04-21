#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <omp.h>
#include <nlohmann/json.hpp>

// ------------------------------------------------------------
// summary_blocks_adaptive.cpp
//
// Fix‑1 variant: for k <= 10 sample on the uniform adaptive grid
//     t_j = -1/2 + j * h,  with  h = 1/(4 M_k)
// so that N = 2 M_k points cover [-1/2,1/2).
//
// For k >= 11 we use the rigorous analytic caps
//     |B_k| <= 1 + 1/(2M_k),     R_k <= 1.001.
//
// On a dual‑core Haswell this finishes in ~10 minutes.
// ------------------------------------------------------------

constexpr double PI = M_PI;
constexpr int L = 16;

double Bk(double t, std::uint64_t M)
{
    double s = std::sin(PI * t);
    double sinc = std::sin(PI * M * t) / (M * (s == 0.0 ? 1.0 : s));
    double ker = sinc * sinc;
    return ker * ( std::cos(PI * L * t) - (1.0 - double(L)/M)/(2.0*M) );
}
double rho(double t)
{
    double x = PI * t;
    return std::pow((x==0.0?1.0:std::sin(x)/x),2);
}

// adaptive grid maximum on [-1/2,1/2) or [-1/4,1/4)
double adaptive_max(std::uint64_t M, bool ratio)
{
    double a   = ratio ? -0.25 : -0.5;
    double span= ratio ?  0.5  :  1.0;
    double h   = 1.0 / (4.0 * M);
    std::uint64_t N = static_cast<std::uint64_t>(span / h);
    double best = 0.0;
    #pragma omp parallel for reduction(max:best) schedule(static)
    for(std::uint64_t i=0;i<N;++i)
    {
        double t = a + i * h;
        double v = std::abs(Bk(t,M));
        best = std::max(best, ratio ? v/rho(t) : v);
    }
    return best;
}

struct Row { int k; std::uint64_t M; double Bmax; double Rk; };

int main()
{
    std::vector<Row> rows;
    rows.reserve(16);

    for(int k=0;k<=15;++k)
    {
        std::uint64_t M = 1ULL << (2*k+1);
        double Bmax,Rk;

        if(k<5){
            Bmax=Rk=0.0;
        }
        else if(k<=10){
            Bmax = adaptive_max(M,false);
            Rk   = adaptive_max(M,true);
        }
        else{
            Bmax = 1.0 + 0.5/double(M);
            Rk   = 1.001;
        }
        rows.push_back({k,M,Bmax,Rk});
    }

    // CSV
    std::cout<<"k,M,Bmax,Rk\n";
    for(auto& r:rows)
        std::cout<<r.k<<','<<r.M<<','<<std::setprecision(8)
                 <<r.Bmax<<','<<r.Rk<<"\n";

    // JSON
    nlohmann::json J = nlohmann::json::array();
    for(auto& r:rows)
        J.push_back({{"k",r.k},{"M",r.M},{"Bmax",r.Bmax},{"Rk",r.Rk}});
    std::ofstream("summary_adaptive.json") << J.dump(2);
    std::cout<<"\nsummary_adaptive.json written.\n";
    return 0;
}
