/* The same scheme again, compiled, with the whole time loop inside C.
 *
 * What this exists to test
 * ------------------------
 * NumPy and PyTorch both express one time step as a sequence of whole-array operations --
 * roughly forty passes over memory per step, each one reading and writing arrays far larger
 * than any cache. On a problem where the arithmetic is about one operation per byte moved,
 * that is the cost. Fusing the step into two sweeps -- one for velocity, one for pressure --
 * cuts the traffic to about 48 bytes per cell per step in single precision, and the question
 * this file answers is how much of the hardware's bandwidth that actually buys.
 *
 * Two sweeps and not one: the pressure update needs the *updated* velocity across its whole
 * stencil, so there is a genuine dependency between them. Fusing further needs temporal
 * blocking, which is a different piece of work.
 *
 * Threads: pthreads, not OpenMP
 * -----------------------------
 * OpenMP would be the obvious choice and it was the first one tried. It does not survive
 * contact with PyTorch: torch ships its own copy of libomp, a second runtime in the same
 * process aborts with "found libomp.dylib already initialized", and the documented workaround
 * is a flag that admits it "may silently produce incorrect results". Since both backends are
 * loaded together in the test suite and in any honest benchmark, that is not a tradeoff worth
 * making. A small persistent pthread pool is about sixty lines, has no external dependency at
 * all -- no libomp to install before the library will build -- and cannot conflict with
 * anyone.
 *
 * Precision is a compile-time choice
 * ----------------------------------
 * REAL is set by the build (-DREAL=float or -DREAL=double) and the library is compiled twice.
 * That keeps one copy of the algorithm rather than two that can drift, and it lets single
 * precision actually halve the memory traffic, which is the entire point of offering it.
 *
 * The struct is a contract
 * ------------------------
 * `acfdtd` is mirrored field for field by a ctypes Structure on the Python side. The two must
 * agree exactly, so `acfdtd_struct_size` exists purely so a test can check that they still do.
 */

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#ifndef REAL
#define REAL double
#endif

/* Oxygen, nitrogen, and the classical term's surrogate. One spare. */
#define MAX_RELAXATION 4
#define MAX_THREADS 64

typedef struct {
    int32_t nx, ny, nz;
    int32_t n_relax;
    int64_t n_wall;
    int64_t n_sources;
    int64_t n_receivers;
    int64_t n_steps;
    int64_t source_length;

    REAL vel_coef;    /* dt / (rho dx) */
    REAL pres_coef;   /* rho c^2 dt / dx */
    REAL src_coef;    /* rho c^2 dt */
    REAL relax_scale; /* dt * sum of the relaxation gains */

    REAL *p;
    REAL *vx;
    REAL *vy;
    REAL *vz;
    REAL *psi; /* n_relax planes of nx*ny*nz, one after another */

    REAL relax_decay[MAX_RELAXATION];
    REAL relax_mean[MAX_RELAXATION];
    REAL relax_gain[MAX_RELAXATION];

    /* Absorbing-layer profiles. `cell` is sampled at cell centres and `face` at faces; a
     * velocity component uses `face` on its own axis and `cell` on the other two. All ones
     * when there is no layer, which costs three multiplies per cell and no memory traffic. */
    const REAL *cell[3];
    const REAL *face[3];
    int32_t has_layer;

    const int64_t *wall_index;
    const REAL *wall_from_updated;
    const REAL *wall_from_previous;
    REAL *wall_scratch;

    const int64_t *source_index;
    const REAL *source_signal; /* n_sources rows of source_length */

    const int64_t *receiver_index;
    REAL *recording; /* n_receivers rows of n_steps */
} acfdtd;

size_t acfdtd_struct_size(void) { return sizeof(acfdtd); }

int acfdtd_is_double(void) { return sizeof(REAL) == sizeof(double); }

/* ---------------------------------------------------------------- the two sweeps ------- */

/* Velocity, with the layer damping folded in.
 *
 * Each cell owns the three faces on its lower side, which is what makes this one loop nest
 * instead of three. The loop bounds, not a conditional, are what hold the wall faces at zero:
 * face 0 on an axis is simply never visited. */
static void update_velocity(const acfdtd *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny, nz = s->nz;
    const REAL c = s->vel_coef;

    for (int32_t i = i_start; i < i_end; i++) {
        for (int32_t j = 0; j < ny; j++) {
            const int64_t cell = ((int64_t)i * ny + j) * nz;
            const REAL *p = s->p + cell;
            const REAL fx = s->cell[0][i], fy = s->cell[1][j];

            REAL *vz = s->vz + ((int64_t)i * ny + j) * (nz + 1);
            const REAL gz = fx * fy;
            for (int32_t k = 1; k < nz; k++)
                vz[k] = (vz[k] - c * (p[k] - p[k - 1])) * (gz * s->face[2][k]);

            if (j >= 1) {
                REAL *vy = s->vy + ((int64_t)i * (ny + 1) + j) * nz;
                const REAL *pj = p - nz;
                const REAL gy = fx * s->face[1][j];
                for (int32_t k = 0; k < nz; k++)
                    vy[k] = (vy[k] - c * (p[k] - pj[k])) * (gy * s->cell[2][k]);
            }

            if (i >= 1) {
                REAL *vx = s->vx + ((int64_t)i * ny + j) * nz;
                const REAL *pi = p - (int64_t)ny * nz;
                const REAL gx = s->face[0][i] * fy;
                for (int32_t k = 0; k < nz; k++)
                    vx[k] = (vx[k] - c * (p[k] - pi[k])) * (gx * s->cell[2][k]);
            }
        }
    }
}

/* Pressure, including the relaxation states. No layer damping here: the wall correction has to
 * see the undamped result, so damping is a separate pass and only when a layer exists. */
static void update_pressure(const acfdtd *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny, nz = s->nz;
    const int64_t n_cells = (int64_t)s->nx * ny * nz;
    const REAL coef = s->pres_coef;
    const int32_t n_relax = s->n_relax;

    for (int32_t i = i_start; i < i_end; i++) {
        for (int32_t j = 0; j < ny; j++) {
            const int64_t cell = ((int64_t)i * ny + j) * nz;
            REAL *p = s->p + cell;
            const REAL *vx = s->vx + cell;
            const REAL *vx_next = s->vx + cell + (int64_t)ny * nz;
            const REAL *vy = s->vy + ((int64_t)i * (ny + 1) + j) * nz;
            const REAL *vy_next = vy + nz;
            const REAL *vz = s->vz + ((int64_t)i * ny + j) * (nz + 1);

            for (int32_t k = 0; k < nz; k++) {
                const REAL divergence =
                    (vx_next[k] - vx[k]) + (vy_next[k] - vy[k]) + (vz[k + 1] - vz[k]);

                REAL added = s->relax_scale * divergence;
                for (int32_t r = 0; r < n_relax; r++) {
                    REAL *state = s->psi + (int64_t)r * n_cells + cell;
                    const REAL target = s->relax_gain[r] * divergence;
                    const REAL offset = state[k] - target;
                    state[k] = target + s->relax_decay[r] * offset;
                    added += s->relax_mean[r] * offset;
                }
                p[k] += added - coef * divergence;
            }
        }
    }
}

static void damp_pressure(const acfdtd *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny, nz = s->nz;

    for (int32_t i = i_start; i < i_end; i++) {
        for (int32_t j = 0; j < ny; j++) {
            REAL *p = s->p + ((int64_t)i * ny + j) * nz;
            const REAL g = s->cell[0][i] * s->cell[1][j];
            for (int32_t k = 0; k < nz; k++) p[k] *= g * s->cell[2][k];
        }
    }
}

/* ------------------------------------------------------- the pressure-only variant ----- */

/* The same physics with half the fields.
 *
 * Eliminating velocity from the staggered pair gives the second-order wave equation on
 * pressure alone, stepped over three time levels:
 *
 *     p^{n+1} = 2 p^n - p^{n-1} + (c dt / dx)^2 * laplacian(p^n)
 *
 * It is not an approximation of the pressure-velocity scheme, it *is* that scheme with the
 * intermediate variable removed -- same Courant limit, same dispersion relation, and starting
 * from `p^{-1} = p^0` it reproduces a run with `v^{-1/2} = 0` to the last bit.
 *
 * The point of it is memory traffic. Two cell-sized arrays instead of four, and with the new
 * level written over the oldest one there are three array touches per cell per step against
 * the pressure-velocity form's twelve. On a problem where bandwidth is the whole cost, that is
 * the largest single lever left.
 *
 * What it gives up is everything the first-order system made easy: the impedance wall, the
 * absorbing layer and the relaxation states would all need re-deriving as three-level
 * relations. This kernel exists to measure what that work would be worth, not to replace
 * anything, so it implements rigid walls only. A rigid wall is a mirrored ghost cell, which is
 * what clamping the neighbour offset to zero does below -- and it is exactly what falls out of
 * eliminating a wall face held at zero velocity. */

typedef struct {
    int32_t nx, ny, nz;
    int64_t n_steps;
    REAL lambda_squared; /* (c dt / dx)^2 */
    REAL *current;
    REAL *previous;
} acfdtd_scalar;

size_t acfdtd_scalar_struct_size(void) { return sizeof(acfdtd_scalar); }

static void scalar_sweep(const acfdtd_scalar *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny, nz = s->nz;
    const int64_t stride_x = (int64_t)ny * nz;
    const int64_t stride_y = nz;
    const REAL scale = s->lambda_squared;

    for (int32_t i = i_start; i < i_end; i++) {
        const int64_t low_x = (i > 0) ? -stride_x : 0;
        const int64_t high_x = (i < s->nx - 1) ? stride_x : 0;

        for (int32_t j = 0; j < ny; j++) {
            const int64_t low_y = (j > 0) ? -stride_y : 0;
            const int64_t high_y = (j < ny - 1) ? stride_y : 0;

            const int64_t base = (int64_t)i * stride_x + (int64_t)j * stride_y;
            const REAL *cur = s->current + base;
            REAL *out = s->previous + base; /* the oldest level, overwritten by the newest */

            /* The transverse part of the stencil, which does not depend on k. */
            const int64_t offsets[4] = {low_x, high_x, low_y, high_y};

            /* The two z faces are peeled out so the inner loop carries no boundary test. */
            for (int32_t k = 0; k < nz; k += (nz > 1 ? nz - 1 : 1)) {
                const REAL neighbours = cur[k + offsets[0]] + cur[k + offsets[1]] +
                                        cur[k + offsets[2]] + cur[k + offsets[3]] +
                                        cur[k + ((k > 0) ? -1 : 0)] +
                                        cur[k + ((k < nz - 1) ? 1 : 0)];
                out[k] = 2 * cur[k] - out[k] + scale * (neighbours - 6 * cur[k]);
            }

            for (int32_t k = 1; k < nz - 1; k++) {
                const REAL neighbours = cur[k + offsets[0]] + cur[k + offsets[1]] +
                                        cur[k + offsets[2]] + cur[k + offsets[3]] + cur[k - 1] +
                                        cur[k + 1];
                out[k] = 2 * cur[k] - out[k] + scale * (neighbours - 6 * cur[k]);
            }
        }
    }
}

/* ------------------------------------------------------------------- the thread pool --- */

/* One pool, shared by every kernel. A sweep is a function plus the range of planes to run it
 * over, so adding a kernel does not mean adding a thread pool or a case to a switch. */
typedef void (*sweep_function)(const void *state, int32_t i_start, int32_t i_end);

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t work_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t work_done = PTHREAD_COND_INITIALIZER;
static pthread_t workers[MAX_THREADS];

static int n_threads = 0;   /* workers including the calling thread */
static int outstanding = 0; /* helper threads still working on this sweep */
static uint64_t generation = 0;
static int stopping = 0;
static const void *shared_state = NULL;
static sweep_function shared_sweep = NULL;
static int32_t shared_extent = 0;

static void velocity_sweep(const void *state, int32_t start, int32_t end) {
    update_velocity((const acfdtd *)state, start, end);
}

static void pressure_sweep(const void *state, int32_t start, int32_t end) {
    update_pressure((const acfdtd *)state, start, end);
}

static void damping_sweep(const void *state, int32_t start, int32_t end) {
    damp_pressure((const acfdtd *)state, start, end);
}

static void wave_sweep(const void *state, int32_t start, int32_t end) {
    scalar_sweep((const acfdtd_scalar *)state, start, end);
}

/* An equal share of the slowest axis each, so every thread owns one contiguous block of
 * memory. Dynamic block-stealing was tried here and measured no better: the kernel saturates
 * the memory system somewhere around eight threads, so what limits a sweep is bandwidth rather
 * than one thread finishing late. The simpler split stays. */
static void run_chunk(const void *state, sweep_function sweep, int32_t extent, int index) {
    const int32_t start = (int32_t)((int64_t)extent * index / n_threads);
    const int32_t end = (int32_t)((int64_t)extent * (index + 1) / n_threads);
    if (start < end) sweep(state, start, end);
}

static void *worker_main(void *argument) {
    const int index = (int)(intptr_t)argument;

    /* Starts at zero, and `generation` is reset to zero whenever the pool is rebuilt. Both
     * halves of that matter, and each one alone is a bug:
     *
     * Without the reset, a rebuilt pool's workers inherit a counter that already looks like
     * pending work, and each runs a chunk immediately against the *previous* run's state
     * pointer -- freed memory by then, and a bus error some distance from the cause.
     *
     * Reading the current generation here instead would fix that and introduce something
     * worse: a worker that finishes starting up after a sweep has been dispatched would decide
     * it had already done that sweep, never decrement the counter, and leave the dispatching
     * thread waiting for it forever. */
    uint64_t seen = 0;

    for (;;) {
        pthread_mutex_lock(&pool_lock);
        while (generation == seen && !stopping) pthread_cond_wait(&work_ready, &pool_lock);
        if (stopping) {
            pthread_mutex_unlock(&pool_lock);
            return NULL;
        }
        seen = generation;
        const void *state = shared_state;
        const sweep_function sweep = shared_sweep;
        const int32_t extent = shared_extent;
        pthread_mutex_unlock(&pool_lock);

        run_chunk(state, sweep, extent, index);

        pthread_mutex_lock(&pool_lock);
        if (--outstanding == 0) pthread_cond_signal(&work_done);
        pthread_mutex_unlock(&pool_lock);
    }
}

static void stop_pool(void) {
    if (n_threads <= 1) {
        n_threads = 0;
        return;
    }
    pthread_mutex_lock(&pool_lock);
    stopping = 1;
    pthread_cond_broadcast(&work_ready);
    pthread_mutex_unlock(&pool_lock);
    for (int index = 1; index < n_threads; index++) pthread_join(workers[index], NULL);

    pthread_mutex_lock(&pool_lock);
    stopping = 0;
    outstanding = 0;
    generation = 0;
    shared_state = NULL;
    pthread_mutex_unlock(&pool_lock);
    n_threads = 0;
}

/* How many threads to use when the caller does not say.
 *
 * Not simply every logical processor. On a machine with both performance and efficiency cores
 * -- which is every recent Apple one -- the measured throughput on all twenty-four logical
 * processors was below what sixteen performance cores alone reached, because the sweep ends
 * when its slowest share does. Where the OS distinguishes the two kinds, use the fast ones;
 * everywhere else this falls through to the processor count and behaves as expected. */
static int default_threads(void) {
#ifdef __APPLE__
    int fast = 0;
    size_t size = sizeof(fast);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &fast, &size, NULL, 0) == 0 && fast > 0)
        return fast;
#endif
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    return online > 0 ? (int)online : 1;
}

/* Set the thread count; 0 asks for the default above. Returns what was set. */
int acfdtd_set_threads(int requested) {
    if (requested <= 0) requested = default_threads();
    if (requested > MAX_THREADS) requested = MAX_THREADS;
    if (requested == n_threads) return n_threads;

    stop_pool();
    n_threads = requested;
    for (int index = 1; index < n_threads; index++)
        pthread_create(&workers[index], NULL, worker_main, (void *)(intptr_t)index);
    return n_threads;
}

int acfdtd_thread_count(void) {
    if (n_threads == 0) acfdtd_set_threads(0);
    return n_threads;
}

/* The calling thread takes chunk 0 itself rather than waiting: one fewer context switch per
 * sweep, and it keeps the serial case free of any pool machinery at all. */
static void dispatch(const void *state, sweep_function sweep, int32_t extent) {
    if (n_threads <= 1) {
        run_chunk(state, sweep, extent, 0);
        return;
    }

    pthread_mutex_lock(&pool_lock);
    shared_state = state;
    shared_sweep = sweep;
    shared_extent = extent;
    outstanding = n_threads - 1;
    generation++;
    pthread_cond_broadcast(&work_ready);
    pthread_mutex_unlock(&pool_lock);

    run_chunk(state, sweep, extent, 0);

    pthread_mutex_lock(&pool_lock);
    while (outstanding > 0) pthread_cond_wait(&work_done, &pool_lock);
    pthread_mutex_unlock(&pool_lock);
}

/* ------------------------------------------------------------------------ the loops ---- */

static void gather_walls(const acfdtd *s) {
    for (int64_t w = 0; w < s->n_wall; w++) s->wall_scratch[w] = s->p[s->wall_index[w]];
}

/* The closed-form solve of the time-centred wall condition, on boundary cells only. */
static void finish_walls(const acfdtd *s) {
    for (int64_t w = 0; w < s->n_wall; w++) {
        const int64_t index = s->wall_index[w];
        s->p[index] = s->p[index] * s->wall_from_updated[w] -
                      s->wall_scratch[w] * s->wall_from_previous[w];
    }
}

void acfdtd_run(acfdtd *s, int64_t step_offset) {
    if (n_threads == 0) acfdtd_set_threads(0);

    for (int64_t step = 0; step < s->n_steps; step++) {
        const int64_t absolute = step_offset + step;

        dispatch(s, velocity_sweep, s->nx);

        if (s->n_wall) gather_walls(s);
        dispatch(s, pressure_sweep, s->nx);
        if (s->n_wall) finish_walls(s);
        if (s->has_layer) dispatch(s, damping_sweep, s->nx);

        for (int64_t source = 0; source < s->n_sources; source++)
            if (absolute < s->source_length)
                s->p[s->source_index[source]] +=
                    s->src_coef * s->source_signal[source * s->source_length + absolute];

        for (int64_t receiver = 0; receiver < s->n_receivers; receiver++)
            s->recording[receiver * s->n_steps + step] = s->p[s->receiver_index[receiver]];
    }
}

/* Leaves the newest level in `current`, having swapped the two pointers once per step. The
 * caller tracks which of its two arrays that is by counting steps. */
void acfdtd_scalar_run(acfdtd_scalar *s) {
    if (n_threads == 0) acfdtd_set_threads(0);

    for (int64_t step = 0; step < s->n_steps; step++) {
        dispatch(s, wave_sweep, s->nx);
        REAL *swap = s->current;
        s->current = s->previous;
        s->previous = swap;
    }
}

/* ============================================================= the 2D kernel ============ */

/* The plane counterpart of the pressure-velocity kernel above: two velocity components instead
 * of three, the inner (contiguous) axis is y, and the cell measure is an area. The derivation,
 * the stored-state pairing (p^n, v^{n-1/2}), and the wall / layer / relaxation treatment are all
 * identical to the 3D kernel and to ac_fdtd.scheme2d; only the indexing loses an axis. The struct
 * is mirrored field for field by a ctypes Structure, and acfdtd2d_struct_size lets a test check
 * that the two still agree. It shares the one thread pool, so there is nothing new below the
 * sweeps. */

typedef struct {
    int32_t nx, ny;
    int32_t n_relax;
    int64_t n_wall;
    int64_t n_sources;
    int64_t n_receivers;
    int64_t n_steps;
    int64_t source_length;

    REAL vel_coef;
    REAL pres_coef;
    REAL src_coef;
    REAL relax_scale;

    REAL *p;
    REAL *vx;
    REAL *vy;
    REAL *psi; /* n_relax planes of nx*ny */

    REAL relax_decay[MAX_RELAXATION];
    REAL relax_mean[MAX_RELAXATION];
    REAL relax_gain[MAX_RELAXATION];

    const REAL *cell[2];
    const REAL *face[2];
    int32_t has_layer;

    const int64_t *wall_index;
    const REAL *wall_from_updated;
    const REAL *wall_from_previous;
    REAL *wall_scratch;

    const int64_t *source_index;
    const REAL *source_signal;

    const int64_t *receiver_index;
    REAL *recording;

    /* Rigid interior obstacles. A velocity face touching a solid cell is held at zero every step
     * (v.n = 0, the same condition that makes the domain's own walls rigid), and a solid cell's
     * pressure is held at zero. The two masks carry that as 0/1 factors so the sweeps stay
     * branchless: `vx_open`/`vy_open` are the open-face factors on the two velocity grids, and
     * `p_keep` is zero on solid cells. All absent when `has_rigid` is 0, which costs nothing. */
    const REAL *vx_open; /* (nx+1)*ny: 0 on x-faces touching a solid, else 1 */
    const REAL *vy_open; /* nx*(ny+1): 0 on y-faces touching a solid, else 1 */
    const REAL *p_keep;  /* nx*ny: 0 on solid cells, else 1 */
    int32_t has_rigid;

    /* Prescribed (hard) sources — the counterpart of add_prescribed_source in the reference scheme,
     * the building block of a shaped velocity/pressure source. Each one overwrites a node inside its
     * window [start, start+pres_length): a pressure source sets p; a velocity source sets vx or vy
     * (its axis). Indices are flat offsets into the matching array. The signals are rows of
     * pres_length. Timing matches the reference: velocity before the pressure update reads it,
     * pressure as the last write before the rigid-cell clamp. */
    int64_t n_pres_pressure;
    int64_t n_pres_velocity;
    int64_t pres_length;
    const int64_t *pp_index;
    const int64_t *pp_start;
    const REAL *pp_signal; /* n_pres_pressure rows of pres_length */
    const int32_t *pv_axis;
    const int64_t *pv_index;
    const int64_t *pv_start;
    const REAL *pv_signal; /* n_pres_velocity rows of pres_length */
} acfdtd2d;

size_t acfdtd2d_struct_size(void) { return sizeof(acfdtd2d); }

/* Velocity, layer damping folded in. vx owns the lower face on the outer axis (i), vy the lower
 * face on the inner axis (j); the loop bounds hold the four wall faces at zero without a test. */
static void update_velocity_2d(const acfdtd2d *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny;
    const REAL c = s->vel_coef;

    for (int32_t i = i_start; i < i_end; i++) {
        const REAL fx = s->cell[0][i];
        const REAL *p = s->p + (int64_t)i * ny;

        REAL *vy = s->vy + (int64_t)i * (ny + 1);
        for (int32_t j = 1; j < ny; j++)
            vy[j] = (vy[j] - c * (p[j] - p[j - 1])) * (fx * s->face[1][j]);

        if (i >= 1) {
            REAL *vx = s->vx + (int64_t)i * ny;
            const REAL *pi = p - ny;
            const REAL gx = s->face[0][i];
            for (int32_t j = 0; j < ny; j++)
                vx[j] = (vx[j] - c * (p[j] - pi[j])) * (gx * s->cell[1][j]);
        }
    }
}

/* Pressure, including the relaxation states; no layer damping here, for the same reason as 3D. */
static void update_pressure_2d(const acfdtd2d *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny;
    const int64_t n_cells = (int64_t)s->nx * ny;
    const REAL coef = s->pres_coef;
    const int32_t n_relax = s->n_relax;

    for (int32_t i = i_start; i < i_end; i++) {
        const int64_t row = (int64_t)i * ny;
        REAL *p = s->p + row;
        const REAL *vx = s->vx + row;
        const REAL *vx_next = s->vx + row + ny;
        const REAL *vy = s->vy + (int64_t)i * (ny + 1);

        for (int32_t j = 0; j < ny; j++) {
            const REAL divergence = (vx_next[j] - vx[j]) + (vy[j + 1] - vy[j]);

            REAL added = s->relax_scale * divergence;
            for (int32_t r = 0; r < n_relax; r++) {
                REAL *state = s->psi + (int64_t)r * n_cells + row;
                const REAL target = s->relax_gain[r] * divergence;
                const REAL offset = state[j] - target;
                state[j] = target + s->relax_decay[r] * offset;
                added += s->relax_mean[r] * offset;
            }
            p[j] += added - coef * divergence;
        }
    }
}

static void damp_pressure_2d(const acfdtd2d *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny;
    for (int32_t i = i_start; i < i_end; i++) {
        REAL *p = s->p + (int64_t)i * ny;
        const REAL g = s->cell[0][i];
        for (int32_t j = 0; j < ny; j++) p[j] *= g * s->cell[1][j];
    }
}

/* Hold the velocity at zero on every face touching a solid cell, after the velocity update and
 * before the pressure update reads it. Multiplying by the 0/1 open-face masks is v.n = 0 on those
 * faces and a no-op everywhere else; it follows the write pattern of update_velocity_2d so a thread
 * touches only its own block. The boundary faces are already zero, so masking them is harmless. */
static void mask_velocity_2d(const acfdtd2d *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny;
    for (int32_t i = i_start; i < i_end; i++) {
        REAL *vy = s->vy + (int64_t)i * (ny + 1);
        const REAL *vyo = s->vy_open + (int64_t)i * (ny + 1);
        for (int32_t j = 0; j < ny + 1; j++) vy[j] *= vyo[j];

        if (i >= 1) {
            REAL *vx = s->vx + (int64_t)i * ny;
            const REAL *vxo = s->vx_open + (int64_t)i * ny;
            for (int32_t j = 0; j < ny; j++) vx[j] *= vxo[j];
        }
    }
}

/* Hold the pressure at zero on solid cells, as the last field operation of the step. With every
 * bounding face already zeroed this is what the divergence would give anyway; it is applied
 * explicitly so the recorded state has exact zeros inside solids. */
static void zero_rigid_pressure_2d(const acfdtd2d *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny;
    for (int32_t i = i_start; i < i_end; i++) {
        REAL *p = s->p + (int64_t)i * ny;
        const REAL *keep = s->p_keep + (int64_t)i * ny;
        for (int32_t j = 0; j < ny; j++) p[j] *= keep[j];
    }
}

static void velocity_sweep_2d(const void *state, int32_t start, int32_t end) {
    update_velocity_2d((const acfdtd2d *)state, start, end);
}
static void pressure_sweep_2d(const void *state, int32_t start, int32_t end) {
    update_pressure_2d((const acfdtd2d *)state, start, end);
}
static void damping_sweep_2d(const void *state, int32_t start, int32_t end) {
    damp_pressure_2d((const acfdtd2d *)state, start, end);
}
static void mask_velocity_sweep_2d(const void *state, int32_t start, int32_t end) {
    mask_velocity_2d((const acfdtd2d *)state, start, end);
}
static void zero_pressure_sweep_2d(const void *state, int32_t start, int32_t end) {
    zero_rigid_pressure_2d((const acfdtd2d *)state, start, end);
}

static void gather_walls_2d(const acfdtd2d *s) {
    for (int64_t w = 0; w < s->n_wall; w++) s->wall_scratch[w] = s->p[s->wall_index[w]];
}

static void finish_walls_2d(const acfdtd2d *s) {
    for (int64_t w = 0; w < s->n_wall; w++) {
        const int64_t index = s->wall_index[w];
        s->p[index] = s->p[index] * s->wall_from_updated[w] -
                      s->wall_scratch[w] * s->wall_from_previous[w];
    }
}

/* Overwrite the prescribed velocity faces for this step, after the velocity update and before the
 * pressure update reads them (a driven piston). Serial: the count is small and scattered. */
static void apply_prescribed_velocity_2d(const acfdtd2d *s, int64_t absolute) {
    for (int64_t k = 0; k < s->n_pres_velocity; k++) {
        const int64_t offset = absolute - s->pv_start[k];
        if (offset >= 0 && offset < s->pres_length) {
            REAL *v = (s->pv_axis[k] == 0) ? s->vx : s->vy;
            v[s->pv_index[k]] = s->pv_signal[k * s->pres_length + offset];
        }
    }
}

/* Overwrite the prescribed pressure nodes, after the volume sources and before the rigid clamp. */
static void apply_prescribed_pressure_2d(const acfdtd2d *s, int64_t absolute) {
    for (int64_t k = 0; k < s->n_pres_pressure; k++) {
        const int64_t offset = absolute - s->pp_start[k];
        if (offset >= 0 && offset < s->pres_length)
            s->p[s->pp_index[k]] = s->pp_signal[k * s->pres_length + offset];
    }
}

void acfdtd2d_run(acfdtd2d *s, int64_t step_offset) {
    if (n_threads == 0) acfdtd_set_threads(0);

    for (int64_t step = 0; step < s->n_steps; step++) {
        const int64_t absolute = step_offset + step;

        dispatch(s, velocity_sweep_2d, s->nx);
        if (s->n_pres_velocity) apply_prescribed_velocity_2d(s, absolute);
        if (s->has_rigid) dispatch(s, mask_velocity_sweep_2d, s->nx);

        if (s->n_wall) gather_walls_2d(s);
        dispatch(s, pressure_sweep_2d, s->nx);
        if (s->n_wall) finish_walls_2d(s);
        if (s->has_layer) dispatch(s, damping_sweep_2d, s->nx);

        for (int64_t source = 0; source < s->n_sources; source++)
            if (absolute < s->source_length)
                s->p[s->source_index[source]] +=
                    s->src_coef * s->source_signal[source * s->source_length + absolute];

        if (s->n_pres_pressure) apply_prescribed_pressure_2d(s, absolute);
        if (s->has_rigid) dispatch(s, zero_pressure_sweep_2d, s->nx);

        for (int64_t receiver = 0; receiver < s->n_receivers; receiver++)
            s->recording[receiver * s->n_steps + step] = s->p[s->receiver_index[receiver]];
    }
}

/* ================================================= the pressure-only 2D kernel =========== */

/* The plane counterpart of acfdtd_scalar: the pressure-velocity 2D scheme with the velocity
 * eliminated, so two cell arrays instead of three and one sweep instead of two. It is the same
 * scheme, not an approximation -- started from previous == current (v = 0) it reproduces
 * AcousticFDTD2D to the last bit -- and it mirrors ac_fdtd.scalar2d.ScalarFDTD2D: a Neumann
 * (mirrored-ghost) Laplacian for rigid walls, closed faces next to a solid cell for obstacles, and
 * the three-level locally-reacting wall relation (1+a) p^{n+1} = 2 p^n - (1-a) p^{n-1} + lam^2 L_N,
 * applied as a post-pass on the wall cells. Open boundaries (the CFS-CPML shell) are not here yet;
 * this kernel is the fast path for a closed room. */
typedef struct {
    int32_t nx, ny;
    int64_t n_wall;
    int64_t n_sources;
    int64_t n_receivers;
    int64_t n_steps;
    int64_t source_length;
    REAL lambda_squared; /* (c dt / dx)^2 */
    REAL src_coef;       /* rho c^2 dt */

    REAL *current;
    REAL *previous; /* the oldest level, overwritten with the newest; pointers swap each step */

    /* Rigid obstacles (0/1 face-open factors and a solid-cell mask), all NULL when has_rigid is 0. */
    const REAL *x_open; /* (nx-1)*ny: 0 on x-faces touching a solid, else 1 */
    const REAL *y_open; /* nx*(ny-1): 0 on y-faces touching a solid, else 1 */
    const REAL *p_keep; /* nx*ny: 0 on solid cells, else 1 */
    int32_t has_rigid;

    /* Locally-reacting walls, as the three-level relation. wall_a is a = c dt xi / (2 dx), summed
     * over the walls at a corner; wall_scratch saves p^{n-1} at the wall cells before the sweep. */
    const int64_t *wall_index;
    const REAL *wall_a;
    REAL *wall_scratch;

    const int64_t *source_index;
    const REAL *source_signal; /* n_sources rows of source_length */

    const int64_t *receiver_index;
    REAL *recording; /* n_receivers rows of n_steps */
} acfdtd2d_scalar;

size_t acfdtd2d_scalar_struct_size(void) { return sizeof(acfdtd2d_scalar); }

/* One sweep writes the new level into `previous` from `current` (p^n) and `previous` (p^{n-1}).
 * The y axis is contiguous. The rigid-wall Neumann condition is the boundary test on each face
 * contributing nothing off the grid; obstacles close individual interior faces via the masks. */
static void scalar_sweep_2d(const acfdtd2d_scalar *s, int32_t i_start, int32_t i_end) {
    const int32_t nx = s->nx, ny = s->ny;
    const REAL scale = s->lambda_squared;

    for (int32_t i = i_start; i < i_end; i++) {
        const int64_t row = (int64_t)i * ny;
        const REAL *cur = s->current + row;
        REAL *out = s->previous + row;
        const REAL *cur_lo = cur - ny; /* row i-1 */
        const REAL *cur_hi = cur + ny; /* row i+1 */

        if (!s->has_rigid) {
            for (int32_t j = 0; j < ny; j++) {
                REAL lap = 0;
                if (i > 0) lap += cur_lo[j] - cur[j];
                if (i < nx - 1) lap += cur_hi[j] - cur[j];
                if (j > 0) lap += cur[j - 1] - cur[j];
                if (j < ny - 1) lap += cur[j + 1] - cur[j];
                out[j] = 2 * cur[j] - out[j] + scale * lap;
            }
        } else {
            const REAL *xo_lo = s->x_open + (int64_t)(i - 1) * ny; /* face (i-1,i) */
            const REAL *xo_hi = s->x_open + row;                   /* face (i,i+1) */
            const REAL *yo = s->y_open + (int64_t)i * (ny - 1);    /* faces (j,j+1) */
            for (int32_t j = 0; j < ny; j++) {
                REAL lap = 0;
                if (i > 0) lap += xo_lo[j] * (cur_lo[j] - cur[j]);
                if (i < nx - 1) lap += xo_hi[j] * (cur_hi[j] - cur[j]);
                if (j > 0) lap += yo[j - 1] * (cur[j - 1] - cur[j]);
                if (j < ny - 1) lap += yo[j] * (cur[j + 1] - cur[j]);
                out[j] = 2 * cur[j] - out[j] + scale * lap;
            }
        }
    }
}

static void zero_rigid_pressure_scalar_2d(const acfdtd2d_scalar *s, int32_t i_start, int32_t i_end) {
    const int32_t ny = s->ny;
    for (int32_t i = i_start; i < i_end; i++) {
        REAL *p = s->previous + (int64_t)i * ny;
        const REAL *keep = s->p_keep + (int64_t)i * ny;
        for (int32_t j = 0; j < ny; j++) p[j] *= keep[j];
    }
}

static void scalar_sweep_2d_dispatch(const void *state, int32_t start, int32_t end) {
    scalar_sweep_2d((const acfdtd2d_scalar *)state, start, end);
}
static void zero_scalar_pressure_dispatch_2d(const void *state, int32_t start, int32_t end) {
    zero_rigid_pressure_scalar_2d((const acfdtd2d_scalar *)state, start, end);
}

void acfdtd2d_scalar_run(acfdtd2d_scalar *s, int64_t step_offset) {
    if (n_threads == 0) acfdtd_set_threads(0);

    for (int64_t step = 0; step < s->n_steps; step++) {
        const int64_t absolute = step_offset + step;

        /* Save p^{n-1} at the wall cells before the sweep overwrites `previous`. */
        for (int64_t w = 0; w < s->n_wall; w++) s->wall_scratch[w] = s->previous[s->wall_index[w]];

        dispatch(s, scalar_sweep_2d_dispatch, s->nx);

        /* Three-level wall post-pass: out_wall = (out_rigid + a p^{n-1}) / (1 + a). */
        for (int64_t w = 0; w < s->n_wall; w++) {
            const int64_t index = s->wall_index[w];
            const REAL a = s->wall_a[w];
            s->previous[index] = (s->previous[index] + a * s->wall_scratch[w]) / (1 + a);
        }

        /* Soft volume sources: the DIFFERENCE S_n - S_{n-1} (the eliminated divergence carries the
         * previous sample). Written into the new level, matching the reference's order. */
        for (int64_t src = 0; src < s->n_sources; src++) {
            if (absolute < s->source_length) {
                const REAL *sig = s->source_signal + src * s->source_length;
                const REAL now = sig[absolute];
                const REAL prev = (absolute > 0) ? sig[absolute - 1] : (REAL)0;
                s->previous[s->source_index[src]] += s->src_coef * (now - prev);
            }
        }

        if (s->has_rigid) dispatch(s, zero_scalar_pressure_dispatch_2d, s->nx);

        /* Swap so `current` holds the newest level, then record from it. */
        REAL *swap = s->current;
        s->current = s->previous;
        s->previous = swap;

        for (int64_t rec = 0; rec < s->n_receivers; rec++)
            s->recording[rec * s->n_steps + step] = s->current[s->receiver_index[rec]];
    }
}
