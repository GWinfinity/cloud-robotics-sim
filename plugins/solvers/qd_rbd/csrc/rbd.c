/* Batch RNEA in plain C, over the exact numpy buffers `qd_rbd.Data` already owns.
 *
 * This exists because REPORT.md §11 killed the SIMD idea: the Quadrants backend
 * emits no vector code and no FP contraction, so the only way to find out what
 * the hardware can actually do with this kernel is to write it somewhere that
 * does both.  The arithmetic below is a transliteration of
 * `algorithms._make_pass` -- same Featherstone block form, same order of
 * operations -- so the two implementations can be diffed to rounding error.
 *
 * What C buys that the traced kernel cannot:
 *   - the per-instance state (R, p, V, A, F for every joint) stays in a stack
 *     frame instead of going through the S buffer;
 *   - model constants are `const double *restrict`, so they are hoisted out of
 *     the batch loop and kept in registers across instances;
 *   - gcc contracts a*b+c into fma (`-ffp-contract=fast` is the default at
 *     -O3) and fuses sin/cos into one `sincos` call.  Taichi's IR has neither.
 *
 * The K parameter is the interesting one.  One instance is a 7-deep dependency
 * chain, so a single instance leaves the machine idle: measured 186 cycles per
 * joint against a throughput bound near 75.  Running K instances through the
 * joint loop *together* (the k loop is inside the i loop, every iteration
 * independent) gives the out-of-order core K independent chains to interleave.
 * That is the same idea as packing 4 arms into one SIMD lane, but expressed as
 * ILP -- and unlike the SIMD version it is something the compiler will actually
 * do once the loop structure permits it.
 */

#define _GNU_SOURCE
#include <math.h>
#include <stddef.h>
#include <string.h>

#define MAXNJ 64

/* ---------------------------------------------------- toolchain portability
 * The kernel is plain C11 plus three GNU-isms: sincos, always_inline and the
 * GCC unroll pragma.  Map them so every supported toolchain keeps the same
 * arithmetic (MSVC just loses fusion/inlining hints, nothing more).
 */
#if defined(_MSC_VER)
static void qd_sincos_impl(double x, double *s, double *c)
{
    *s = sin(x);
    *c = cos(x);
}
#define qd_sincos qd_sincos_impl
#define QD_ALWAYS_INLINE static __forceinline
#define QD_UNROLL8
#define QD_OMP_FOR
#else
#define qd_sincos sincos
#define QD_ALWAYS_INLINE static inline __attribute__((always_inline))
#if defined(__GNUC__) || defined(__clang__)
#define QD_UNROLL8 _Pragma("GCC unroll 8")
#else
#define QD_UNROLL8
#endif
#define QD_OMP_FOR _Pragma("omp parallel for schedule(static) if(B >= 32)")
#endif

typedef struct {
    int nj, nv;
    const int    *par;    /* nj      parent index                        */
    const int    *kind;   /* nj      0 revolute, 1 prismatic, 2 fixed    */
    const int    *iv;     /* nj      dof index, -1 for a fixed joint     */
    const double *axis;   /* 3*nj    unit axis, joint frame              */
    const double *Rpl;    /* 9*nj    parent->joint rotation, row major   */
    const double *ppl;    /* 3*nj    parent->joint translation           */
    const double *A0;     /* 9*nj    body inertia upper-left block       */
    const double *B0;     /* 9*nj    body inertia right column block     */
    const double *mass;   /* nj                                          */
    double grav[3];
} rbd_model;

/* ---------------------------------------------------------------- 3-vectors */
static inline void cross3(const double *a, const double *b, double *c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

static inline void mv3(const double *M, const double *x, double *y)
{
    y[0] = M[0] * x[0] + M[1] * x[1] + M[2] * x[2];
    y[1] = M[3] * x[0] + M[4] * x[1] + M[5] * x[2];
    y[2] = M[6] * x[0] + M[7] * x[1] + M[8] * x[2];
}

static inline void mTv3(const double *M, const double *x, double *y)
{
    y[0] = M[0] * x[0] + M[3] * x[1] + M[6] * x[2];
    y[1] = M[1] * x[0] + M[4] * x[1] + M[7] * x[2];
    y[2] = M[2] * x[0] + M[5] * x[1] + M[8] * x[2];
}

static inline void mm3(const double *A, const double *B, double *C)
{
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

/* Rodrigues, axis assumed unit length.  Matches spatial.rot_axis_angle. */
static inline void rot_aa(const double *ax, double ang, double *R)
{
    double s, c;
#ifdef QD_RBD_NOTRIG          /* timing probe only: destroys the rotation */
    (void)ang; s = 0.0; c = 1.0;
#else
    qd_sincos(ang, &s, &c);
#endif
    const double t = 1.0 - c;
    const double x = ax[0], y = ax[1], z = ax[2];
    const double tx = t * x, ty = t * y, tz = t * z;
    const double sx = s * x, sy = s * y, sz = s * z;
    R[0] = tx * x + c;  R[1] = tx * y - sz;  R[2] = tx * z + sy;
    R[3] = tx * y + sz; R[4] = ty * y + c;   R[5] = ty * z - sx;
    R[6] = tx * z - sy; R[7] = ty * z + sx;  R[8] = tz * z + c;
}

/* ------------------------------------------------------- per-instance state */
typedef struct {
    double R[MAXNJ][9], p[MAXNJ][3];
    double V1[MAXNJ][3], V2[MAXNJ][3], A1[MAXNJ][3], A2[MAXNJ][3];
    double F1[MAXNJ][3], F2[MAXNJ][3];
} rnea_state;

static inline void rnea_base(const rbd_model *m, rnea_state *st)
{
    memset(st->V1[0], 0, sizeof st->V1[0]);
    memset(st->V2[0], 0, sizeof st->V2[0]);
    memset(st->A1[0], 0, sizeof st->A1[0]);
    st->A2[0][0] = -m->grav[0];
    st->A2[0][1] = -m->grav[1];
    st->A2[0][2] = -m->grav[2];
}

QD_ALWAYS_INLINE void
rnea_fwd(const rbd_model *restrict m, rnea_state *restrict st, const int i,
         const double *restrict q, const double *restrict qd,
         const double *restrict qdd, const int use_acc)
{
    const double *Rp = m->Rpl + 9 * i;
    const double *pp = m->ppl + 3 * i;
    const double *ax = m->axis + 3 * i;
    const int kind = m->kind[i];
    const int iv = m->iv[i];
    const int par = m->par[i];

    if (kind == 2) {                       /* fixed joint */
        memcpy(st->R[i], Rp, 9 * sizeof(double));
        st->p[i][0] = pp[0]; st->p[i][1] = pp[1]; st->p[i][2] = pp[2];
    } else {
        const double qi = q[iv];
        if (kind == 0) {                   /* revolute */
            double Rr[9];
            rot_aa(ax, qi, Rr);
            mm3(Rp, Rr, st->R[i]);
            st->p[i][0] = pp[0]; st->p[i][1] = pp[1]; st->p[i][2] = pp[2];
        } else {                           /* prismatic */
            memcpy(st->R[i], Rp, 9 * sizeof(double));
            st->p[i][0] = pp[0] + qi * (Rp[0] * ax[0] + Rp[1] * ax[1] + Rp[2] * ax[2]);
            st->p[i][1] = pp[1] + qi * (Rp[3] * ax[0] + Rp[4] * ax[1] + Rp[5] * ax[2]);
            st->p[i][2] = pp[2] + qi * (Rp[6] * ax[0] + Rp[7] * ax[1] + Rp[8] * ax[2]);
        }
    }

    const double *Ri = st->R[i];
    const double *pi = st->p[i];
    double *v1 = st->V1[i], *v2 = st->V2[i], *a1 = st->A1[i], *a2 = st->A2[i];
    double tmp[3], c1[3];

    /* motion transform parent -> child:  v' = (E v1, E (v2 - p x v1)) */
    cross3(pi, st->V1[par], tmp);
    mTv3(Ri, st->V1[par], v1);
    tmp[0] = st->V2[par][0] - tmp[0];
    tmp[1] = st->V2[par][1] - tmp[1];
    tmp[2] = st->V2[par][2] - tmp[2];
    mTv3(Ri, tmp, v2);

    cross3(pi, st->A1[par], tmp);
    mTv3(Ri, st->A1[par], a1);
    tmp[0] = st->A2[par][0] - tmp[0];
    tmp[1] = st->A2[par][1] - tmp[1];
    tmp[2] = st->A2[par][2] - tmp[2];
    mTv3(Ri, tmp, a2);

    /* joint motion subspace, then the velocity cross-product term */
    if (kind == 0) {
        const double vdi = qd[iv];
        c1[0] = ax[0] * vdi; c1[1] = ax[1] * vdi; c1[2] = ax[2] * vdi;
        if (use_acc) {
            const double qddi = qdd[iv];
            a1[0] += ax[0] * qddi; a1[1] += ax[1] * qddi; a1[2] += ax[2] * qddi;
        }
        v1[0] += c1[0]; v1[1] += c1[1]; v1[2] += c1[2];
        cross3(v1, c1, tmp);
        a1[0] += tmp[0]; a1[1] += tmp[1]; a1[2] += tmp[2];
        cross3(v2, c1, tmp);            /* v2 x wj; the uj term is zero */
        a2[0] += tmp[0]; a2[1] += tmp[1]; a2[2] += tmp[2];
    } else if (kind == 1) {
        const double vdi = qd[iv];
        c1[0] = ax[0] * vdi; c1[1] = ax[1] * vdi; c1[2] = ax[2] * vdi;
        if (use_acc) {
            const double qddi = qdd[iv];
            a2[0] += ax[0] * qddi; a2[1] += ax[1] * qddi; a2[2] += ax[2] * qddi;
        }
        v2[0] += c1[0]; v2[1] += c1[1]; v2[2] += c1[2];
        cross3(v1, c1, tmp);            /* v1 x uj */
        a2[0] += tmp[0]; a2[1] += tmp[1]; a2[2] += tmp[2];
    }

    /* f = I a + v x* (I v), body coordinates */
    const double *A0i = m->A0 + 9 * i;
    const double *B0i = m->B0 + 9 * i;
    const double mi = m->mass[i];
    double *f1 = st->F1[i], *f2 = st->F2[i];
    double Iv1[3], Iv2[3];

    mv3(A0i, v1, Iv1);
    mv3(B0i, v2, tmp);
    Iv1[0] += tmp[0]; Iv1[1] += tmp[1]; Iv1[2] += tmp[2];
    mTv3(B0i, v1, Iv2);
    Iv2[0] += mi * v2[0]; Iv2[1] += mi * v2[1]; Iv2[2] += mi * v2[2];

    mv3(A0i, a1, f1);
    mv3(B0i, a2, tmp);
    f1[0] += tmp[0]; f1[1] += tmp[1]; f1[2] += tmp[2];
    cross3(v1, Iv1, tmp);
    f1[0] += tmp[0]; f1[1] += tmp[1]; f1[2] += tmp[2];
    cross3(v2, Iv2, tmp);
    f1[0] += tmp[0]; f1[1] += tmp[1]; f1[2] += tmp[2];

    mTv3(B0i, a1, f2);
    f2[0] = f2[0] + mi * a2[0];
    f2[1] = f2[1] + mi * a2[1];
    f2[2] = f2[2] + mi * a2[2];
    cross3(v1, Iv2, tmp);
    f2[0] += tmp[0]; f2[1] += tmp[1]; f2[2] += tmp[2];
}

QD_ALWAYS_INLINE void
rnea_bwd(const rbd_model *restrict m, rnea_state *restrict st, const int i,
         double *restrict tau)
{
    const int kind = m->kind[i];
    const int iv = m->iv[i];
    if (iv >= 0) {
        const double *ax = m->axis + 3 * i;
        const double *f = (kind == 0) ? st->F1[i] : st->F2[i];
        tau[iv] = ax[0] * f[0] + ax[1] * f[1] + ax[2] * f[2];
    }
    const int par = m->par[i];
    const double *Ri = st->R[i];
    const double *pi = st->p[i];
    double n2[3], n1[3];
    mv3(Ri, st->F2[i], n2);
    mv3(Ri, st->F1[i], n1);
    n1[0] += pi[1] * n2[2] - pi[2] * n2[1];
    n1[1] += pi[2] * n2[0] - pi[0] * n2[2];
    n1[2] += pi[0] * n2[1] - pi[1] * n2[0];
    st->F1[par][0] += n1[0]; st->F1[par][1] += n1[1]; st->F1[par][2] += n1[2];
    st->F2[par][0] += n2[0]; st->F2[par][1] += n2[1]; st->F2[par][2] += n2[2];
}

/* ------------------------------------------------- K instances in one sweep */
#define DEFINE_RNEA(K)                                                         \
static void rnea_batch##K(const rbd_model *restrict m,                          \
                          const double *restrict q,                            \
                          const double *restrict qd,                           \
                          const double *restrict qdd,                          \
                          double *restrict tau, const int B,                   \
                          const int use_acc)                                   \
{                                                                              \
    const int nj = m->nj, nv = m->nv;                                          \
    const int nb = (B / (K)) * (K);                                            \
    QD_OMP_FOR                                                                 \
    for (int b = 0; b < nb; b += (K)) {                                        \
        rnea_state st[K];                                                      \
        QD_UNROLL8                                                             \
        for (int k = 0; k < (K); ++k) rnea_base(m, &st[k]);                    \
        for (int i = 1; i < nj; ++i) {                                         \
            QD_UNROLL8                                                         \
            for (int k = 0; k < (K); ++k) {                                    \
                const size_t off = (size_t)(b + k) * nv;                       \
                rnea_fwd(m, &st[k], i, q + off, qd + off, qdd + off, use_acc); \
            }                                                                  \
        }                                                                      \
        for (int i = nj - 1; i >= 1; --i) {                                    \
            QD_UNROLL8                                                         \
            for (int k = 0; k < (K); ++k)                                      \
                rnea_bwd(m, &st[k], i, tau + (size_t)(b + k) * nv);            \
        }                                                                      \
    }                                                                          \
    for (int b = nb; b < B; ++b) {                                             \
        rnea_state st;                                                         \
        rnea_base(m, &st);                                                     \
        const size_t off = (size_t)b * nv;                                     \
        for (int i = 1; i < nj; ++i)                                           \
            rnea_fwd(m, &st, i, q + off, qd + off, qdd + off, use_acc);        \
        for (int i = nj - 1; i >= 1; --i) rnea_bwd(m, &st, i, tau + off);      \
    }                                                                          \
}

DEFINE_RNEA(1)
DEFINE_RNEA(2)
DEFINE_RNEA(4)
DEFINE_RNEA(8)

/* ------------------------------------------------------------ entry points */
void rbd_rnea_k(const rbd_model *m, const double *q, const double *v,
                const double *a, double *tau, int B, int K)
{
    switch (K) {
    case 1: rnea_batch1(m, q, v, a, tau, B, 1); break;
    case 2: rnea_batch2(m, q, v, a, tau, B, 1); break;
    case 8: rnea_batch8(m, q, v, a, tau, B, 1); break;
    default: rnea_batch4(m, q, v, a, tau, B, 1); break;
    }
}

void rbd_rnea_bias_k(const rbd_model *m, const double *q, const double *v,
                     double *tau, int B, int K)
{
    switch (K) {
    case 1: rnea_batch1(m, q, v, v, tau, B, 0); break;
    case 2: rnea_batch2(m, q, v, v, tau, B, 0); break;
    case 8: rnea_batch8(m, q, v, v, tau, B, 0); break;
    default: rnea_batch4(m, q, v, v, tau, B, 0); break;
    }
}

int rbd_max_joints(void) { return MAXNJ; }
