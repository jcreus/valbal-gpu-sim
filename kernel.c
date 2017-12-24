#define J 8
#define Fs <?Fs?>
#define T <?T?>
#define MAXLS <?local_size?>
#define DEBUG <?debug?>
#define GROUPS_PER_BETARR <?ngroups?>

#define glob (get_global_id(1)*get_global_size(0)+get_global_id(0))
#if DEBUG != 0
#define DARRAY(idx, t, val) debug[T*Fs*DEBUG*glob + T*Fs*idx + t] = val;
#else
#define DARRAY(idx, t, val)
#endif

#include "utils.c"
#include "random.c"
#include "<?pasta?>.c"

void kernel simulate(global const float* args,
					 global const uint2* seeds,
					 global float* output,
					 global float* debug) {
	const int arg0 = (J + K)*get_global_id(0);
	float h = (float)args[arg0];
	float l = args[arg0 + 1];

	double alt_beta[6];
	for (int k=0; k<6; k++) alt_beta[k] = (double)args[arg0 + 2 + k];

	double beta[K];
	for (int k=0; k<K; k++) beta[k] = args[arg0 + J + k];

	uint2 seed = seeds[glob];

	double x_alt[6] = {h,h,h,h,h,h};

	controller_state state;
	init_state(&state);

	__local float rand_buf[MAXLS*2];
	int rndmax = 2*get_local_size(0);
	int rndid = rndmax;

	for (uint t=0; t<(T*Fs); t++) {
		if (rndid == rndmax) {
			float2 rnd = normal(&seed);
			rand_buf[2*get_local_id(0)] = rnd.x;
			rand_buf[2*get_local_id(0)+1] = rnd.y;
			// XXX possibly unroll a bit more
			barrier(CLK_LOCAL_MEM_FENCE);
			rndid = 0;
		}

		float filt = (float)update_biquad(alt_beta, x_alt, h);
		DARRAY(0, t, h);
		DARRAY(1, t, filt);

		l += 0.0002 * rand_buf[rndid++];
		float dl = update_state(&state, beta, h, t, debug);
		DARRAY(2, t, 100000*dl);

		l += dl/Fs;
		float v = 31*l + 0.7 * rand_buf[rndid++];
		h += v/Fs;
	}

	output[glob] = h;
}
