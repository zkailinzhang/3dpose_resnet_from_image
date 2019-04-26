import pickle                                                           
fr = open('0_info.pkl') 

fr = pickle.load(open(r'0_info.pkl','rb'))                              

In [10]: fr                                                                     
Out[10]: 
[array([[ -71,  125],
        [ -88,   64],
        [  30, -126],
        [ -51, -201],
        [ -96,  200],
        [ 207,  -71],
        [  33,  -13],
        [-139,  175],
        [-154,   34],
        [ 167,  148],
        [ -54, -209],
        [ 204,   66],
        [ 236, -266]], dtype=int32),
 array([[[ 0.323951  ,  0.944309  , -0.0577546 ,  0.04307093],
         [ 0.200026  , -0.128031  , -0.97139   ,  0.04470536],
         [-0.924687  ,  0.30313   , -0.230362  ,  0.81350666]],
 
        [[ 0.991305  , -0.0697777 ,  0.111555  ,  0.0741947 ],
         [ 0.0157081 , -0.778987  , -0.626844  , -0.02562867],
         [ 0.13064   ,  0.623146  , -0.771118  ,  0.6714113 ]],
 



import matplotlib.pyplot as plt                                         

In [2]: import matplotlib.image as mpimg                                        

In [3]: import numpy as np                                                      

In [4]:                                                                         

In [4]: lena = mpimg.imread('0_depth.png')                                      

In [5]: lena                                                                    
Out[5]: 
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

In [6]: lena.shape                                                              
Out[6]: (480, 640)

In [7]: plt.show(lena)                                                          

In [8]: plt.imshow(lena)                                                        
Out[8]: <matplotlib.image.AxesImage at 0x7f3b59ef0320>

In [9]: plt.show()      




import numpy as np
import open3d as o3 
from open3d import JVisualizer

pts_path = '../.ply'

fragment  = o3.read_point_cloud(pts_path)

visualizer = JVisualizer()

visualizer.add_geometry(fragment)

visualizer.show() 貌似不行


#或者直接

o3.draw_geometries([fragment]) 


将python print  打印输出到 txt 

print("xxxx",file=open("output.txt","a"))
or
import sys
sys.stdout= open('output.txt','wt')

print("xxxxxx")




ff = open(r'../posedb/cat_real.pkl','rb')

pts= pickle.load(ff)

for k,v in pt[0].items(): 
    ...:     print(k,v) 
    ...:     break 
    ...:      
    ...:                                                                                                                 
RT [[ 0.0950661   0.983309   -0.155129    0.0717302 ]
 [ 0.741596   -0.173913   -0.647911   -0.14907229]
 [-0.664076   -0.0534489  -0.745752    1.0606388 ]]








 trans_init = np.asarray([[0.001,0.0,0.0,0.0],[0.0,0.001,0.0,0.0],[0.0,0.0,0.001,0.0],[0.0,0.0,0.0,1.0]])        

In [92]: trans_init                                                                                                      
Out[92]: 
array([[0.001, 0.   , 0.   , 0.   ],
       [0.   , 0.001, 0.   , 0.   ],
       [0.   , 0.   , 0.001, 0.   ],
       [0.   , 0.   , 0.   , 1.   ]])

In [93]: plydata2.transform(trans_init)                                                                                  

In [94]: o3.draw_geometries([plydata,plydata2])    




diff = cv2.subtract(img2,img1)
result = not np.any(diff)

if result: print("yiyang")




np.load ('a.npy')
np.load('a.npz')




In [135]: for k,v in pts2[0].items(): 
     ...:     print(k) 
     ...:      
     ...:      
     ...:      
     ...:                                                                                                                
rgb_pth
dpt_pth
RT
cls_typ
rnd_typ
corners
farthest
farthest4
farthest12
farthest16
farthest20
center
small_bbox
van_pts
In [145]: len(pts2)                                                                                                      
Out[145]: 1188

In [136]: pts2[0].get('RT')                                                                                              
Out[136]: 
array([[-0.985486  ,  0.00825023, -0.169555  ,  0.0146043 ],
       [ 0.130482  ,  0.675735  , -0.725504  , -0.11608808],
       [ 0.108589  , -0.737098  , -0.667004  ,  1.0171365 ]],
      dtype=float32)






In [16]: np.mgrid[0:7]                                                          
Out[16]: array([0, 1, 2, 3, 4, 5, 6])

In [17]: np.mgrid[0:7,0:2]                                                      
Out[17]: 
array([[[0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6]],

       [[0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]]])

In [18]: np.mgrid[0:7,0:2].shape                                                
Out[18]: (2, 7, 2)

In [19]: np.mgrid[0:7,0:3].shape                                                
Out[19]: (2, 7, 3)




In [20]: np.mgrid[0:7,0:6].T                                                    
Out[20]: 
array([[[0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [6, 0]],

       [[0, 1],
        [1, 1],
        [2, 1],
        [3, 1],
        [4, 1],
        [5, 1],
        [6, 1]],

       [[0, 2],
        [1, 2],
        [2, 2],
        [3, 2],
        [4, 2],
        [5, 2],
        [6, 2]],

       [[0, 3],
        [1, 3],
        [2, 3],
        [3, 3],
        [4, 3],
        [5, 3],
        [6, 3]],

       [[0, 4],
        [1, 4],
        [2, 4],
        [3, 4],
        [4, 4],
        [5, 4],
        [6, 4]],

       [[0, 5],
        [1, 5],
        [2, 5],
        [3, 5],
        [4, 5],
        [5, 5],
        [6, 5]]])

In [21]: np.mgrid[0:7,0:6]                                                      
Out[21]: 
array([[[0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6, 6]],

       [[0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5]]])
