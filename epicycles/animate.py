import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_curve(points : np.ndarray, error : float, step=10, interval=30):
    """
    points : np.ndarray
        복소수로 표현된 곡선의 점들
    error : float
        상대오차
    step : int
        애니메이션의 단계
    interval : int
        애니메이션의 간격
    """
    xs, ys = points.real, points.imag
    K = len(points)
    fig, ax = plt.subplots()
    m = 0.05 * max(np.ptp(xs), np.ptp(ys))      
    ax.set_xlim(xs.min() - m, xs.max() + m)     
    ax.set_ylim(ys.min() - m, ys.max() + m)
    ax.set_aspect("equal")
    ax.set_title(f"error: {error:.4f}")
    line, = ax.plot([], [], ".", markersize=2)   

    def update(num):
        line.set_data(xs[:num], ys[:num])        
        return line,

    frames = list(range(step, K, step)) + [K]    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)                               # 노트북에서 그림이 한 번 더 뜨는 것 방지
    return ani

