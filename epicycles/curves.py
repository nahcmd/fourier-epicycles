import numpy as np

def star(t):
    idx = np.arange(11)                           # 꼭짓점 번호 0..10 (V_10 = V_0)
    ang = np.pi/2 + idx*np.pi/5                   # 각도
    r = np.where(idx % 2 == 0, 1.0, 0.382)        # 바깥 1, 안쪽 0.382
    V = r*np.exp(1j*ang)                          # 꼭짓점 (복소수)m
    s = 10*t/(2*np.pi)
    j = np.floor(s).astype(int) % 10              # 몇 번째 변인지
    u = s - np.floor(s)                           # 변 위에서의 비율
    return V[j] + u*(V[j+1] - V[j])
