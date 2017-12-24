#define K 10
#define Tminb 2
#define Tminv 5
#define cmd_max 0.001f
#define cmd_min 0.00001f

typedef struct controller_state_s {
	uint tlastb;
	uint tlastv;
	double filter[6];
} controller_state;

inline void init_state(controller_state* state) {
	state->tlastb = 0;
	state->tlastv = 0;
	for (int k=0; k<6; k++) state->filter[k] = 0;
}

/* Beta:
 *  - 0-5 6 biquad constants
 *  - 6 gain factor
 *  - 7 dlb
 *  - 8 dlv
 *  - 9 stp
 */

inline float update_state(controller_state* state, double* beta, float h, uint t, global float* debug) {
	double effort = update_biquad(beta, state->filter, beta[9]-h);
	//DARRAY(3, t, effort);
	float dlcmd = effort*beta[6];
	//DARRAY(4, t, dlcmd);
	dlcmd = (fabs(dlcmd) < cmd_max)*dlcmd + (fabs(dlcmd) >= cmd_max) * sign(dlcmd)*cmd_max;
	//DARRAY(5, t, dlcmd);
	dlcmd = step(cmd_min, fabs(dlcmd))*dlcmd;
	//DARRAY(6, t, dlcmd);
	uint Twaitb = 86400;
	if (dlcmd > 0) Twaitb = fabs((float)beta[7]*Tminb/dlcmd);
	uint Twaitv = 86400;
	if (dlcmd < 0) Twaitv = fabs((float)beta[8]*Tminv/dlcmd);
	//DARRAY(7, t, Twaitb);
	//DARRAY(8, t, Twaitv);
	//DARRAY(9, t,((dlcmd < 0.0f) * fabs((float)beta[8]*Tminv/dlcmd) + (dlcmd >= 0.0f)*86400));
	if (t-state->tlastb >= Twaitb) state->tlastb = t;
	if (t-state->tlastv >= Twaitv) state->tlastv = t;

	return (t-state->tlastb < Tminb)*beta[7] + (t-state->tlastv < Tminv)*beta[8];
}
