#include "stencil/solve.h"
#include <assert.h>
#include <math.h>

void solve_jacobi(mesh_t *A, mesh_t const *B, mesh_t *C)
{
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;

    usz index_c, index_plus_o, index_minus_o;
    f64 power;

    usz const block_size_x = 50;
    usz const block_size_y = 50;
    usz const block_size_z = 50;

    for (usz i_block = STENCIL_ORDER; i_block < dim_x - STENCIL_ORDER; i_block += block_size_x)
    {
        for (usz j_block = STENCIL_ORDER; j_block < dim_y - STENCIL_ORDER; j_block += block_size_y)
        {
            for (usz k_block = STENCIL_ORDER; k_block < dim_z - STENCIL_ORDER; k_block += block_size_z)
            {
                // Itération à l'intérieur des blocs
                for (usz i = i_block; i < fmin(i_block + block_size_x, dim_x - STENCIL_ORDER); ++i)
                {
                    for (usz j = j_block; j < fmin(j_block + block_size_y, dim_y - STENCIL_ORDER); ++j)
                    {
                        for (usz k = k_block; k < fmin(k_block + block_size_z, dim_z - STENCIL_ORDER); ++k)
                        {
                            index_c = dim_x * dim_y * i + dim_x * j + k;
                            C->cells[index_c] = A->cells[index_c] * B->cells[index_c];

                            for (usz o = 1; o <= STENCIL_ORDER; ++o)
                            {
                                power = pow(17.0, (f64)o);

                                index_plus_o = dim_x * dim_y * i + dim_x * j + k + o;
                                index_minus_o = dim_x * dim_y * i + dim_x * j + k - o;

                                C->cells[index_c] += A->cells[index_plus_o] * B->cells[index_plus_o] / power;
                                C->cells[index_c] += A->cells[index_minus_o] * B->cells[index_minus_o] / power;

                                index_plus_o = dim_x * dim_y * i + dim_x * (j + o) + k;
                                index_minus_o = dim_x * dim_y * i + dim_x * (j - o) + k;

                                C->cells[index_c] += A->cells[index_plus_o] * B->cells[index_plus_o] / power;
                                C->cells[index_c] += A->cells[index_minus_o] * B->cells[index_minus_o] / power;

                                index_plus_o = dim_x * dim_y * (i + o) + dim_x * j + k;
                                index_minus_o = dim_x * dim_y * (i - o) + dim_x * j + k;

                                C->cells[index_c] += A->cells[index_plus_o] * B->cells[index_plus_o] / power;
                                C->cells[index_c] += A->cells[index_minus_o] * B->cells[index_minus_o] / power;
                            }
                        }
                    }
                }
            }
        }
    }

    mesh_copy_core(A, C);
}

