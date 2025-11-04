import numpy as np
import matplotlib.pyplot as plt

def function(tt , f):
    t = np.arange(tt)
    return np.cos(np.pi * 2 * 800 * t / f) + np.cos(np.pi * 2 * 888.8 * t / f) + np.cos(np.pi * 2 * 1600 * t / f)

for i in range(3):
    fig, [(graph1, graph2), (graph3, graph4), (graph5, graph6), (graph7, graph8), (graph9, grapha)] = plt.subplots(nrows = 5, ncols = 2)

    graph1.stem(np.arange(64),function(64,8000))
    graph2.plot(np.arange(64),np.abs(np.fft.fft(function(64,8000),64)))
    graph3.plot(np.arange(1024),np.abs(np.fft.fft(function(64,8000),1024)))
    graph4.plot(np.arange(2048),np.abs(np.fft.fft(function(64,8000),2048)))
    graph5.plot(np.arange(256),np.abs(np.fft.fft(function(256,8000))))
    graph6.plot(np.arange(512),np.abs(np.fft.fft(function(512,8000))))
    graph7.plot(np.arange(1024),np.abs(np.fft.fft(function(256,8000),1024)))
    graph8.plot(np.arange(2048),np.abs(np.fft.fft(function(256,8000),2048)))
    graph9.plot(np.arange(1024),np.abs(np.fft.fft(function(512,8000),1024)))
    grapha.plot(np.arange(2048),np.abs(np.fft.fft(function(512,8000),2048)))

    plt.show()