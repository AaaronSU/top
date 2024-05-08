#include "stencil/solve.h"
#include <assert.h>
#include <math.h>
#include <omp.h>

#define min(x, y) (((x) <= (y)) * (x) + ((x) > (y)) * (y))

static const f64 power_list[8] = {
    pow(17.0, 1),
    pow(17.0, 2),
    pow(17.0, 3),
    pow(17.0, 4),
    pow(17.0, 5),
    pow(17.0, 6),
    pow(17.0, 7),
    pow(17.0, 8),
};

void solve_jacobi(mesh_t *A, mesh_t const *B, mesh_t *C)
{
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;

    usz const block_size_x = 100;
    usz const block_size_y = 100;
    usz const block_size_z = 100;

#pragma omp parallel for

    for (usz i_block = STENCIL_ORDER; i_block < dim_x - STENCIL_ORDER; i_block += block_size_x)
    {
        for (usz j_block = STENCIL_ORDER; j_block < dim_y - STENCIL_ORDER; j_block += block_size_y)
        {
            for (usz k_block = STENCIL_ORDER; k_block < dim_z - STENCIL_ORDER; k_block += block_size_z)
            {
                usz min_x = min(i_block + block_size_x, dim_x - STENCIL_ORDER);
                usz min_y = min(j_block + block_size_y, dim_y - STENCIL_ORDER);
                usz min_z = min(k_block + block_size_z, dim_z - STENCIL_ORDER);
                // Itération à l'intérieur des blocs
                for (usz i = i_block; i < min_x; ++i)
                {
                    for (usz j = j_block; j < min_y; ++j)
                    {
                        usz dim_xy = dim_x * dim_y;
                        for (usz k = k_block; k < min_z; ++k)
                        {
                            usz indice = dim_xy * i + dim_y * j + k;
                            C->cells[indice] =
                                A->cells[indice] * B->cells[indice];
                            for (usz o = 1; o <= STENCIL_ORDER; ++o)
                            {
                                C->cells[indice] +=
                                    (A->cells[indice + dim_xy * o] * B->cells[indice + dim_xy * o] +
                                     A->cells[indice - dim_xy * o] * B->cells[indice - dim_xy * o] +
                                     A->cells[indice + dim_y * o] * B->cells[indice + dim_y * o] +
                                     A->cells[indice - dim_y * o] * B->cells[indice - dim_y * o] +
                                     A->cells[indice + o] * B->cells[indice + o] +
                                     A->cells[indice - o] * B->cells[indice - o]) /
                                    power_list[o - 1];
                            }
                        }
                    }
                }
            }
        }
    }

    mesh_copy_core(A, C);
}
