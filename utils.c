inline double update_biquad(double* k, double* x, double inp) {
	// (x0, x1, x2, y0, y1, y2)
	// (a0, a1, a2, b0, b1, b2)
	// (0 , 1 , 2 , 3 , 4 , 5 )
	x[0] = x[1];
	x[1] = x[2];
	x[3] = x[4];
	x[4] = x[5];
	x[2] = inp;
	x[5] = (1.f/k[0])*(k[3]*x[2] + k[4]*x[1] + k[5]*x[0] - k[1]*x[4] - k[2]*x[3]);
	
	//x[5] = k[3]*x[2] + k[4]*x[1] + k[5]*x[0];
	//double helper = -k[1]*x[4] - k[2]*x[3];

	//x[5] /= k[0];
	return x[5];
}
