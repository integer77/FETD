/* Minimal sequential MUMPS wrapper: complex symmetric LDL^T in single (c) or double (z) precision. */
#include <stdlib.h>
#include <string.h>
#include "cmumps_c.h"
#include "zmumps_c.h"

typedef struct { int prec; void *id; } handle;

void *mf_create(int prec, int n, long nz, int *irn, int *jcn, void *a, int *info)
{
    handle *h = malloc(sizeof *h);
    h->prec = prec;
    if (prec == 0) {
        CMUMPS_STRUC_C *id = calloc(1, sizeof *id);
        id->comm_fortran = -987654; id->par = 1; id->sym = 2; id->job = -1;
        cmumps_c(id);
        id->icntl[0] = id->icntl[1] = id->icntl[2] = -1; id->icntl[3] = 0;
        id->icntl[13] = 60;
        id->n = n; id->nnz = nz; id->irn = irn; id->jcn = jcn; id->a = a;
        id->job = 4; cmumps_c(id);
        info[0] = id->infog[0]; info[1] = id->infog[28]; info[2] = id->infog[18]; info[3] = id->infog[12];
        h->id = id;
    } else {
        ZMUMPS_STRUC_C *id = calloc(1, sizeof *id);
        id->comm_fortran = -987654; id->par = 1; id->sym = 2; id->job = -1;
        zmumps_c(id);
        id->icntl[0] = id->icntl[1] = id->icntl[2] = -1; id->icntl[3] = 0;
        id->icntl[13] = 60;
        id->n = n; id->nnz = nz; id->irn = irn; id->jcn = jcn; id->a = a;
        id->job = 4; zmumps_c(id);
        info[0] = id->infog[0]; info[1] = id->infog[28]; info[2] = id->infog[18]; info[3] = id->infog[12];
        h->id = id;
    }
    return h;
}

int mf_solve(void *hp, void *rhs)
{
    handle *h = hp;
    if (h->prec == 0) { CMUMPS_STRUC_C *id = h->id; id->rhs = rhs; id->job = 3; cmumps_c(id); return id->infog[0]; }
    ZMUMPS_STRUC_C *id = h->id; id->rhs = rhs; id->job = 3; zmumps_c(id); return id->infog[0];
}

void mf_free(void *hp)
{
    handle *h = hp;
    if (h->prec == 0) { CMUMPS_STRUC_C *id = h->id; id->job = -2; cmumps_c(id); free(id); }
    else { ZMUMPS_STRUC_C *id = h->id; id->job = -2; zmumps_c(id); free(id); }
    free(h);
}
