import numpy as np
import matplotlib.pyplot as plt

n = np.arange(32)
x = np.cos(2 * np.pi * n / np.sqrt(31))
y = np.fft.fft(x)
y_512 = np.fft.fft(x, 512) 

b = np.blackman(32)
y_b = np.fft.fft(b, 512)
x_b = b * x
y_x_b = np.fft.fft(x_b,512)

h = np.hamming(32)
y_h = np.fft.fft(h, 512)
x_h = h * x
y_x_h = np.fft.fft(x_h,512)


fig, [(graph0, graph1), (graph2, graph3), (graph4, graph5)] = plt.subplots(nrows = 3, ncols = 2)
graph0.plot(n/16,x)
graph1.stem(n/16,x)
graph2.stem(n/16,np.abs(y))
graph3.semilogx(n/16,np.abs(y))
graph4.plot(np.arange(512),np.abs(y_512))
graph5.semilogx(np.arange(512),np.abs(y_512))
plt.show()

fig2, [(graph_0, graph_1), (graph_2, graph_3), (graph_4, graph_5), (graph_6, graph_7), (graph_8, graph_9)] = plt.subplots(nrows = 5, ncols = 2)
graph_0.plot(n/16,b)
graph_1.plot(n/16,h)
graph_2.plot(np.arange(512),np.abs(y_b))
graph_3.plot(np.arange(512),np.abs(y_h))
graph_4.semilogx(np.arange(512),np.abs(y_b))
graph_5.semilogx(np.arange(512),np.abs(y_h))
graph_6.plot(n/16,x_b)
graph_7.plot(n/16,x_h)
graph_8.plot(np.arange(512),np.abs(y_x_b))
graph_9.plot(np.arange(512),np.abs(y_x_h))
plt.show()
"""
Matlab kod
clc
n = 0:31;
x = cos(2*pi*n/sqrt(31));
%subplot(211),stem(n,x)
m = 0:31;
y = abs(fft(x));
%subplot(211),stem(2*m/32,y)
%subplot(212),semilogy(2*m/32,y),grid on

w = blackman(32);
f = w .* x';
subplot(221),plot(0:31,w)git pull
subplot(222),plot(0:31,f)
v = abs(fft(f,512));
h = 0:511;
subplot(223),stem(h/16,v)
subplot(224),semilogy(h/16,v)

%a = hamming(32) .* x';
%b = abs(fft(a,512));
%c = 0:511;
%subplot(223),stem(c/16,b)
%subplot(224),semilogy(c/16,b)
"""