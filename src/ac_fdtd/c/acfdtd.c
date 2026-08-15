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

/* ------------------------------------------------------------------- the thread pool --- */

enum { SWEEP_VELOCITY, SWEEP_PRESSURE, SWEEP_DAMP };

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t work_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t work_done = PTHREAD_COND_INITIALIZER;
static pthread_t workers[MAX_THREADS];

static int n_threads = 0;    /* total workers including the calling thread */
static int outstanding = 0;  /* helper threads still working on this sweep */
static uint64_t generation = 0;
static int stopping = 0;
static const acfdtd *shared_state = NULL;
static int shared_sweep = 0;

/* An equal share of the slowest axis each, so every thread owns one contiguous block of memory.
 *
 * Dynamic block-stealing was tried here and measured no better: the kernel saturates the memory
 * system somewhere around eight threads, so what limits a sweep is bandwidth rather than one
 * thread finishing late. The simpler split stays. */
static void run_chunk(const acfdtd *s, int sweep, int index) {
    const int32_t start = (int32_t)((int64_t)s->nx * index / n_threads);
    const int32_t end = (int32_t)((int64_t)s->nx * (index + 1) / n_threads);
    if (start >= end) return;

    switch (sweep) {
        case SWEEP_VELOCITY: update_velocity(s, start, end); break;
        case SWEEP_PRESSURE: update_pressure(s, start, end); break;
        default: damp_pressure(s, start, end); break;
    }
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
        const acfdtd *state = shared_state;
        const int sweep = shared_sweep;
        pthread_mutex_unlock(&pool_lock);

        run_chunk(state, sweep, index);

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
static void dispatch(const acfdtd *s, int sweep) {
    if (n_threads <= 1) {
        run_chunk(s, sweep, 0);
        return;
    }

    pthread_mutex_lock(&pool_lock);
    shared_state = s;
    shared_sweep = sweep;
    outstanding = n_threads - 1;
    generation++;
    pthread_cond_broadcast(&work_ready);
    pthread_mutex_unlock(&pool_lock);

    run_chunk(s, sweep, 0);

    pthread_mutex_lock(&pool_lock);
    while (outstanding > 0) pthread_cond_wait(&work_done, &pool_lock);
    pthread_mutex_unlock(&pool_lock);
}

/* ------------------------------------------------------------------------ the loop ----- */

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

        dispatch(s, SWEEP_VELOCITY);

        if (s->n_wall) gather_walls(s);
        dispatch(s, SWEEP_PRESSURE);
        if (s->n_wall) finish_walls(s);
        if (s->has_layer) dispatch(s, SWEEP_DAMP);

        for (int64_t source = 0; source < s->n_sources; source++)
            if (absolute < s->source_length)
                s->p[s->source_index[source]] +=
                    s->src_coef * s->source_signal[source * s->source_length + absolute];

        for (int64_t receiver = 0; receiver < s->n_receivers; receiver++)
            s->recording[receiver * s->n_steps + step] = s->p[s->receiver_index[receiver]];
    }
}
