import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_work_space(x,y,z):
    a_2 = 431.8
    a_3 = 20.32
    d_2_3 = 149.09
    d_4 = 433.07
    d_6 = 56.25
    sin_1 = -d_2_3/np.sqrt(x**2 + y**2)
    if abs(sin_1) > 1:
        return False
    theta_1 = np.arcsin(sin_1) + np.arctan2(y,x)
    if theta_1 >= 160/180*np.pi:
        theta_1 = -np.pi - np.arcsin(sin_1) + np.arctan2(y,x)
    elif theta_1 <= -160/180*np.pi:
        theta_1 = np.pi - np.arcsin(sin_1) + np.arctan2(y,x)

    sin_2 = (a_2**2 + z**2 + (np.cos(theta_1)*x+np.sin(theta_1)*y)**2 \
                - (d_4+d_6)**2 - a_3**2) / \
            (2*a_2*np.sqrt(z**2 + (np.cos(theta_1)*x+np.sin(theta_1)*y)**2))
    thi_2 = np.arctan2(z,np.cos(theta_1)*x+np.sin(theta_1)*y)
    theta_2_max = np.pi*1.25+thi_2
    theta_2_max = theta_2_max % (2*np.pi)
    theta_2_min = theta_2_max - np.pi/2*3
    if np.tan(theta_2_max) <= 0:
        sin_2_max = 1
        sin_2_min = -1
    else:
        if theta_2_max < np.pi/2:
            sin_2_max = max(np.sin(theta_2_max),np.sin(theta_2_min))
            sin_2_min = -1
        else:
            sin_2_max = 1
            sin_2_min = min(np.sin(theta_2_max),np.sin(theta_2_min))
    if sin_2 > sin_2_max or sin_2 < sin_2_min:
        return False
    
    cos_3 = (sin_2*np.sqrt(z**2 + (np.cos(theta_1)*x+np.sin(theta_1)*y)**2)\
                - a_2) / \
            np.sqrt((d_4+d_6)**2 + a_3**2)
    thi_3 = np.arctan2(d_4+d_6,a_3)
    theta_3_max = np.pi/4 + thi_3
    theta_3_max = theta_3_max % (2*np.pi)
    theta_3_min = theta_3_max - np.pi/2*3
    if np.tan(theta_3_max) >= 0:
        cos_3_max = 1
        cos_3_min = -1
    else:
        if theta_3_max < np.pi:
            cos_3_max = 1
            cos_3_min = min(np.cos(theta_3_max),np.cos(theta_3_min))
        else:
            cos_3_min = -1
            cos_3_max = max(np.cos(theta_3_max),np.cos(theta_3_min))
    if cos_3 > cos_3_max or cos_3 < cos_3_min:
        return False

    return True
    
def find_work_space():
    a_2 = 431.8
    d_2_3 = 149.09
    d_4 = 433.07
    d_6 = 56.25
    axis_lim = 1000
    interval = 20
    fig = plt.figure()
    ax = Axes3D(fig)
    position = []
    for i in range(-axis_lim,axis_lim,interval):
        for j in range(-axis_lim,axis_lim,interval):
            if d_2_3/np.sqrt(i**2 + j**2) > 1 or np.sqrt(i**2 + j**2) > a_2+d_4+d_6:
                continue 
            last_position = False
            for k in range(-axis_lim,axis_lim,interval):
                if is_work_space(i,j,k) != last_position:
                    position.append([i,j,k])
                    last_position = not last_position
        print(i)
    position = np.array(position)
    np.save('position.npy',position)
    # position = np.load('position.npy')
    ax.scatter(position[:,0],position[:,1],position[:,2],c=position[:,2],s=4)
    plt.show()
    
if __name__ == '__main__':
    find_work_space()
