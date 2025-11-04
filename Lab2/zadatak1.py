import numpy as np
import matplotlib.pyplot as plt
n = np.arange(32)
x = np.cos(2 * np.pi * n / np.sqrt(31))

"""
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
subplot(221),plot(0:31,w)
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