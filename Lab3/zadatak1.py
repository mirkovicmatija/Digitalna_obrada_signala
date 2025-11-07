import numpy as np
import matplotlib.pyplot as plt

def function(tt):
    f = 8000
    t = np.arange(tt)
    return np.cos(np.pi * 2 * 800 * t / f) + np.cos(np.pi * 2 * 888.8 * t / f) + np.cos(np.pi * 2 * 1600 * t / f)

funcs = [function, np.blackman, np.hamming]

for f in funcs:
    fig, [(graph1, graph2), (graph3, graph4), (graph5, graph6), (graph7, graph8), (graph9, grapha)] = plt.subplots(nrows = 5, ncols = 2)

    graph1.stem(np.arange(64),f(64))
    graph2.plot(np.arange(64),np.abs(np.fft.fft(f(64),64)))
    graph3.plot(np.arange(1024),np.abs(np.fft.fft(f(64),1024)))
    graph4.plot(np.arange(2048),np.abs(np.fft.fft(f(64),2048)))
    graph5.plot(np.arange(256),np.abs(np.fft.fft(f(256))))
    graph6.plot(np.arange(512),np.abs(np.fft.fft(f(512))))
    graph7.plot(np.arange(1024),np.abs(np.fft.fft(f(256),1024)))
    graph8.plot(np.arange(2048),np.abs(np.fft.fft(f(256),2048)))
    graph9.plot(np.arange(1024),np.abs(np.fft.fft(f(512),1024)))
    grapha.plot(np.arange(2048),np.abs(np.fft.fft(f(512),2048)))

    plt.show()