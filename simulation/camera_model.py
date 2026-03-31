import numpy as np

def get_K(fx=1000, fy=1000, cx=960, cy=540):   # 1920x1080
    """Returns the 3x3 intrinsic matrix."""
    
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)

def get_camera_pose(rx=0, ry=0, rz=0, tx=0.0, ty=0, tz=0):
    """
    Camera 1 is the identity
    Returns rotation R and translation t for Camera 2.
    """
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    Rx = np.array([[1,0,0],
                   [0,np.cos(rx),-np.sin(rx)],
                   [0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],
                    [0,1,0],
                    [-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                   [np.sin(rz),np.cos(rz),0],
                   [0,0,1]])

    R = Rz @ Ry @ Rx
    t = np.array([tx, ty, tz], dtype=np.float64)
    return R, t