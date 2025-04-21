// generate_minorants_v9_stream64.cpp
// ----------------------------------
// Stream‑writing version with 64‑bit loop counter (Windows‑safe).

#include <gmpxx.h>
#include <mpfr.h>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <cstdint>

static const mpz_class TWO60 = mpz_class(1) << 60;
constexpr int L = 16;

// --------------------------------------------------------------------------
void write_block(int k)
{
    const mpz_class M  = mpz_class(1) << (2 * k + 1);   // 2^(2k+1)
    const uint64_t Ml = M.get_ui();                     // 64‑bit safe

    std::ofstream ofs("block_" + std::to_string(k) + ".json");
    ofs << "{\"k\":" << k
        << ",\"M\":" << Ml
        << ",\"L\":" << L
        << ",\"a_coeffs\":[";

    #pragma omp parallel
    {
        std::string buf; buf.reserve(1 << 20);

        #pragma omp for schedule(static)
        for (uint64_t m = 1; m <= Ml; ++m)
        {
            mpz_class num(0), den(1);
            if (m >= L)
            {
                mpfr_t t; mpfr_init2(t, 256);
                mpfr_const_pi(t, MPFR_RNDZ);
                mpfr_mul_ui(t, t, static_cast<uint64_t>(L) * m, MPFR_RNDZ);
                mpfr_div_ui(t, t, Ml, MPFR_RNDZ);
                mpfr_sin(t, t, MPFR_RNDZ);
                mpfr_mul_ui(t, t, Ml - m, MPFR_RNDZ);
                mpfr_div_ui(t, t, Ml, MPFR_RNDZ);
                mpfr_div_ui(t, t, m,  MPFR_RNDZ);
                mpfr_const_pi(t, MPFR_RNDZ);
                mpfr_div(t, t, t, MPFR_RNDZ);

                mpfr_mul_z(t, t, TWO60.get_mpz_t(), MPFR_RNDZ);
                mpfr_get_z(num.get_mpz_t(), t, MPFR_RNDZ);
                den = TWO60;
                mpz_class g; mpz_gcd(g.get_mpz_t(), num.get_mpz_t(), den.get_mpz_t());
                if (g != 0) { num /= g; den /= g; }
                mpfr_clear(t);
            }
            buf += '[' + std::to_string(m) + ','
                   + num.get_str() + ','
                   + den.get_str() + "],";
            if (buf.size() > (1 << 18)) {
                #pragma omp critical
                ofs << buf, buf.clear();
            }
        }
        if (!buf.empty()) {
            #pragma omp critical
            ofs << buf;
        }
    }

    ofs.seekp(-1, std::ios_base::cur);   // remove last comma
    ofs << "],\"int_B\":[0,1]}";
    ofs.close();
    std::cout << "block_" << k << ".json written\n";
}

// --------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc != 2) { std::cerr << "usage: ./gen_blocks k\n"; return 1; }
    int k = std::stoi(argv[1]);
    if (k < 0 || k > 15) { std::cerr << "k must be 0..15\n"; return 1; }
    write_block(k);
    return 0;
}
