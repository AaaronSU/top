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

    for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; ++k)
    {
        for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; ++j)
        {
            for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; ++i)
            {
                C->cells[dim_x * dim_y * i + dim_y * j + k] = A->cells[dim_x * dim_y * i + dim_y * j + k] * B->cells[dim_x * dim_y * i + dim_y * j + k];

                for (usz o = 1; o <= STENCIL_ORDER; ++o)
                {
                    f64 power = pow(17, (f64)o);
                    C->cells[dim_x * dim_y * i + dim_y * j + k] += A->cells[dim_x * dim_y * (i + o) + dim_y * j + k] *
                                                                   B->cells[dim_x * dim_y * (i + o) + dim_y * j + k] / power;
                    C->cells[dim_x * dim_y * i + dim_y * j + k] += A->cells[dim_x * dim_y * (i - o) + dim_y * j + k] *
                                                                   B->cells[dim_x * dim_y * (i - o) + dim_y * j + k] / power;
                    C->cells[dim_x * dim_y * i + dim_y * j + k] += A->cells[dim_x * dim_y * i + dim_y * (j + o) + k] *
                                                                   B->cells[dim_x * dim_y * i + dim_y * (j + o) + k] / power;
                    C->cells[dim_x * dim_y * i + dim_y * j + k] += A->cells[dim_x * dim_y * i + dim_y * (j - o) + k] *
                                                                   B->cells[dim_x * dim_y * i + dim_y * (j - o) + k] / power;
                    C->cells[dim_x * dim_y * i + dim_y * j + k] += A->cells[dim_x * dim_y * i + dim_y * j + k + o] *
                                                                   B->cells[dim_x * dim_y * i + dim_y * j + k + o] / power;
                    C->cells[dim_x * dim_y * i + dim_y * j + k] += A->cells[dim_x * dim_y * i + dim_y * j + k - o] *
                                                                   B->cells[dim_x * dim_y * i + dim_y * j + k - o] / power;
                }
            }
        }
    }

    mesh_copy_core(A, C);
}
