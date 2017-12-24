inline ushort2 uniformrandom(uint2 *state) {
	// Uniform random numbers adadpted from David B. Thomas'
    enum { A = 4294883355U };
    uint x = (*state).x, c = (*state).y;
    uint res = x^c;
    uint hi = mul_hi(x,A);
    x = x*A + c;
    c = hi + (x<c);
    *state = (uint2)(x,c);
    return *(ushort2*)&res;
}


inline float fastcos(float x) {
    x *= 1.f/(2.f*3.141592653);
    x -= .25f + floor(x + 0.25f);
    x *= 16.f * (fabs((float)x) - 0.5f);
    x += .225f * x * (fabs(x) - 1.f);
    return x;
}


inline float2 normal(uint2* state) {
	/* y i k e s
	 * i k e s y
	 * k e s y i
	 * e s y i k
	 * s y i k e */
	ushort2 z = uniformrandom(state);

	float u1 = 1.52587890625e-05 * (1+z.x);
	float u2 = 9.5873799240e-05*z.y;

	float base = half_sqrt(-2.f*half_log(u1));
	float s = half_sin(u2);
	float c = half_cos(u2);
	return (float2)(base*c, base*s);
}

/*inline float normal(float mu, float sigma, uint2 *state) {
	return mu + sigma*boxmuller(state);
}*/
