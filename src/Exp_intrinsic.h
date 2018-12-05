#include <stdio.h>
#include <immintrin.h>
#include <math.h>
/*    gcc -O3 -m64 -Wall -mavx2 -march=broadwell  expc.c -lm     */

/**
 * @brief Evaluates the function 1 - exp(tau) for tau negative.  Vectorises well.
 *        Accurate to 6.18 digits (single precision) for entire domain (including
 *        near zero, unlike intrinsic)
 *
 *        Valid for tau ~= [-1.5e6, 0]
 * @param tau input
 * @param expv output
 * @param length length of vector
 */
inline void cram7(float x, float* expv) {

  /* Generated in Mathematica, accurate to 6.18 digits (single precision), tau [-1.5e6,0] */
  float c1n = -1.00000014302666667201396424463;
  float c2n = 0.234841040052684510704433796447;
  float c3n = -0.0624785939603762121316592924635;
  float c4n = 0.0100434102711342948752684759736;
  float c5n = -0.00135724435934263932676353754751;
  float c6n = 0.0000951474224366003625378414851577;
  float c7n = -0.0000160076055315534285575266516209;

  float c0d = 1;
  float c1d = -0.734847118148952339633322706422;
  float c2d = 0.263193362386411901729092564316;
  float c3d = -0.0609467155163113059870970359654;
  float c4d = 0.0100863490579686697359577926719;
  float c5d = -0.00135667018708833025497446407598; 
  float c6d = 0.0000951502816434275317085698085885;
  float c7d = -0.0000160076032420105715765981718742;

  float num, den;

    den = c7d;
    den = den * x + c6d;
    den = den * x + c5d;
    den = den * x + c4d;
    den = den * x + c3d;
    den = den * x + c2d;
    den = den * x + c1d;
    den = den * x + c0d;

    num = c7n;
    num = num * x + c6n;
    num = num * x + c5n;
    num = num * x + c4n;
    num = num * x + c3n;
    num = num * x + c2n;
    num = num * x + c1n;
    num = num * x;

    *expv = 1.f - num / den;
}

/**
 * @brief Based on: https://codingforspeed.com/using-faster-exponential-approximation/
 *        with the added knowledge that increasing polynomial order scales faster than
 *        squaring.
 *
 *        Valid for inputs tau ~= [-81, 0] in single precision.
 * @param tau input
 * @param expv output
 * @param length length of vector
 */
inline void newlimit(float x, float* expv) {

  /* Generated in Mathematica, approximates (e^x)^(1/32) */
  float c0 = 1.0;
  float c1 = 0.031249996853940119553;
  float c2 = 0.00048827111506219862388;
  float c3 = 5.0775354594032959870e-6;
  float c4 = 3.7119438192380573565e-8;

  float val;

    val = c4;
    val = val * x + c3;
    val = val * x + c2;
    val = val * x + c1;
    val = val * x + c0;

    val *= val;
    val *= val;
    val *= val;
    val *= val;
    val *= val;

    *expv = val;
}

/**
 * @brief https://codingforspeed.com/using-faster-exponential-approximation/
 *
 *        Valid for inputs tau ~= [-8125, 0] in single precision.
 * @param tau input
 * @param expv output
 * @param length length of vector
 */
inline void limit(float x, float* expv) {
  float c0 = 1.0;
  float c1 = 0.000244140625;
  float val;

    val = c1 * x + c0;

    val *= val;
    val *= val;
    val *= val;
    val *= val;
    
    val *= val;
    val *= val;
    val *= val;
    val *= val;
    
    val *= val;
    val *= val;
    val *= val;
    val *= val;

    *expv = val;
}

/**
 * @brief Built-in function
 * @param tau input
 * @param expv output
 * @param length length of vector
 */
inline void exp_real(float x, float* expv, int length) {
  
    *expv = exp(x);
}


inline void exp256_ps(float x1, float* exp) {
/* Modified code from this source: https://github.com/reyoung/avx_mathfun

   AVX implementation of exp
   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/
   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.
  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:
  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  (this is the zlib license)

*/
/* 
  To increase the compatibility across different compilers the original code is
  converted to plain AVX2 intrinsics code without ingenious macro's,
  gcc style alignment attributes etc.
  Moreover, the part "express exp(x) as exp(g + n*log(2))" has been significantly simplified.
  This modified code is not thoroughly tested!
*/
__m256 x = _mm256_load_ps(&x1);

__m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
__m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

__m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
__m256   inv_LOG2EF    = _mm256_set1_ps(0.693147180559945f);

__m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
__m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
__m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
__m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
__m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
__m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
__m256   fx;
__m256i  imm0;
__m256   one           = _mm256_set1_ps(1.0f);

        x     = _mm256_min_ps(x, exp_hi);
        x     = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
        fx     = _mm256_mul_ps(x, cephes_LOG2EF);
        fx     = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
__m256  z      = _mm256_mul_ps(fx, inv_LOG2EF);
        x      = _mm256_sub_ps(x, z);
        z      = _mm256_mul_ps(x,x);

__m256  y      = cephes_exp_p0;
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p1);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p2);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p3);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p4);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p5);
        y      = _mm256_mul_ps(y, z);
        y      = _mm256_add_ps(y, x);
        y      = _mm256_add_ps(y, one);

  /* build 2^n */
        imm0   = _mm256_cvttps_epi32(fx);
        imm0   = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        imm0   = _mm256_slli_epi32(imm0, 23);
__m256  pow2n  = _mm256_castsi256_ps(imm0);
        y      = _mm256_mul_ps(y, pow2n);
        _mm256_store_ps(exp, y);
}

#include <stdio.h>
#include <stdint.h>
// From Nic Schraudolph https://stackoverflow.com/questions/47025373/fastest-implementation-of-exponential-function-using-sse
inline float fastExp3(register float x)  // cubic spline approximation
{
    union { float f; int32_t i; } reinterpreter;

    reinterpreter.i = (int32_t)(12102203.0f*x) + 127*(1 << 23);
    int32_t m = (reinterpreter.i >> 7) & 0xFFFF;  // copy mantissa
    // empirical values for small maximum relative error (8.34e-5):
    reinterpreter.i +=
         ((((((((1277*m) >> 14) + 14825)*m) >> 14) - 79749)*m) >> 11) - 626;
    return reinterpreter.f;
}

inline float fastExp4(register float x)  // quartic spline approximation
{
    union { float f; int32_t i; } reinterpreter;

    reinterpreter.i = (int32_t)(12102203.0f*x) + 127*(1 << 23);
    int32_t m = (reinterpreter.i >> 7) & 0xFFFF;  // copy mantissa
    // empirical values for small maximum relative error (1.21e-5):
    reinterpreter.i += (((((((((((3537*m) >> 16)
        + 13668)*m) >> 18) + 15817)*m) >> 14) - 80470)*m) >> 11);
    return reinterpreter.f;
}

// http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

// Underflow of exponential is common practice in numerical routines,
// so handle it here.
#include <stdint.h>
#define cast_uint32_t static_cast<uint32_t>

static inline float
fastpow2 (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

static inline float
fastexp (float p)
{
  return fastpow2 (1.442695040f * p);
}

static inline float
expFromInt (float p)
{
  int xb = int(p);
  float r = p - xb;
  float pol = ((((r/5 + 1)*r/4 + 1)*r/3 + 1)*r/2 + 1)*r + 1; 
  return std::pow(2.71828182846f, xb) * pol;
}
