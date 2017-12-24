#!/usr/bin/python

import numpy as np
#np.random.seed(0)
import math
import matplotlib.pyplot as plt
import pyopencl as cl
import os
import time
import random
import functions
import scipy.stats
import socket

if os.environ['USER'] == 'joan' and socket.gethostname() != 'frankie':
    os.environ['PYOPENCL_CTX'] = '2'

default_settings = ('spaghetti', 'marinara', (('T', 20000), ('Fs', 1), ('debug', 0)))

class GPUMC:
    def __init__(self, local_size=32):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.kernels = {}
        with open('kernel.c') as f:
            self.base_code = f.read()

        self.local_size = local_size
        stride = 32
        if local_size % stride != 0:
            print "[OHP] Picking a local size that's not a multiple of %d is a pretty bold move" % stride
            if random.random() > 0.93: # punishment for picking the wrong local size
                print "[TLITC]"
                exit(137)

    def evaluate(self, betas, N, kernel=default_settings):
        if betas.dtype != np.float32:
            print "[OHP] Please use float32s... Doing otherwise adds overhead"
            betas = betas.astype(np.float32)
            if random.random() > 0.91: # punishment for picking the wrong type
                print "[TLITC]"
                exit(137)
        K = betas.shape[0]
        M = betas.shape[1]
        betas.shape = K*M

        groups_for_betas = int(math.ceil(float(K)/self.local_size))
        print "groups required a round of betas", groups_for_betas

        if not kernel in self.kernels:
            code = self.base_code.replace('<?pasta?>', kernel[0])
            for x, y in kernel[2]:
                code = code.replace('<?%s?>' % x, str(y))
            code = code.replace('<?local_size?>', str(self.local_size))
            code = code.replace('<?ngroups?>', str(groups_for_betas))
            self.kernels[kernel] = cl.Program(self.ctx, code).build(["-cl-finite-math-only","-cl-mad-enable"])
            print "Finished cooking",kernel[0],"with",kernel[1]

        mf = cl.mem_flags
        k = self.kernels[kernel].simulate
        d = dict(kernel[2])
        debug = d['debug']
        if debug != 0 and N > 10:
            print "[OHP] Debugging with big N... bold move, I like it."
            exit(137)

        T = d['T']
        Fs = d['Fs']

        args_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=betas)
        k.set_arg(0, args_d)

        S = N*K*2
        rand = np.random.randint(0, 2**32, size=S, dtype=np.uint32)
        rand_d = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rand)
        k.set_arg(1, rand_d)
        
        O = N*K * 20000
        output_d = cl.Buffer(self.ctx, mf.WRITE_ONLY, 4*N*K)
        k.set_arg(2, output_d)

        debug_d = cl.Buffer(self.ctx, mf.WRITE_ONLY, max(4,N*K*T*Fs*debug*4))
        k.set_arg(3, debug_d)

        t0 = time.time()
        cl.enqueue_nd_range_kernel(self.queue, k, [K,N], [self.local_size, 1])
        self.queue.finish()
        print (N*K)/(time.time() - t0), "servings per second"

        output = np.zeros(N*K, dtype=np.float32)
        cl.enqueue_copy(self.queue, output, output_d)
        output.shape = (N, K)
        print output, len(output)
        print np.mean(output), np.std(output)
        #print scipy.stats.mstats.normaltest(output)
        #plt.hist(output)
        #plt.show()

        if debug != 0:
            debug_h = np.zeros(N*K*T*Fs*debug, dtype=np.float32)
            cl.enqueue_copy(self.queue, debug_h, debug_d)
            debug_h.shape = (N*K, debug, T*Fs)

            print debug_h
            for i in range(0, debug):
                plt.plot(debug_h[0][i], label=str(i))
            plt.legend()
            plt.show()

            #print np.mean(output), np.std(output)
        return output

        
        

if __name__ == '__main__':
    sim = GPUMC()
    (a, b) = functions.biquad_lowpass(0.01, 0.5, 1)
    (ap, bp) = [1, -0.30000000000000004, -0.4], [2.8291628469154848e-05, -1.4117522606108269e-05, -1.4131668420342846e-05]
    params = [13500, 0]
    params.extend(sum([a, b, ap, bp], []))
    params.append(0.05) # gain factor
    params.append(0.001) # dlb
    params.append(-0.002) # dlv
    params.append(13500) # stp
    betas = np.array([params for _ in range(64)], dtype=np.float32)
    print betas.shape
    sim.evaluate(betas, 100)
