# graphRank

To do anything with the code first go in the python directory
```
cd python
```

# Usage
You can check all the options of the code using

```
python graphRank.py --help
```

```
usage: graphRank.py [-h] [--testID TESTID] [--trainID TRAINID] [--graph GRAPH]
                    [--check CHECK] [--outfile OUTFILE] [--tune_kernel]
                    [--test] [--lamb LAMB] [--walk WALK] [--func FUNC]
                    [--cuda] [--gpu_block GPU_BLOCK [GPU_BLOCK ...]]

test graphRank

optional arguments:
  -h, --help            show this help message and exit
  --testID TESTID       list of ID for testing. Default: testID.lst
  --trainID TRAINID     list of ID for training. Default: trainID.lst
  --graph GRAPH         folder containing the graph of each complex. Default:
                        graphMAT
  --check CHECK         file containing the kernel. Default:
                        kernelMAT/<testID_name>.mat
  --outfile OUTFILE     Output file containing the calculated Kernel values.
                        Default: kernel.pkl
  --tune_kernel         Only tune the CUDA kernel
  --test                Only test the functions on a single pair pair of graph
  --lamb LAMB           Lambda parameter in the Kernel calculations. Default:
                        1
  --walk WALK           Max walk length in the Kernel calculations. Default: 4
  --method METHOD       Method used in the calculation: 'vect'(default),
                        'combvec', 'iter'
  --func FUNC           functions to tune in the kernel. Defaut: all functions
  --cuda                Use CUDA kernel
  --gpu_block GPU_BLOCK [GPU_BLOCK ...]
                        number of gpu block to use. Default: 8 8 1

```

# Test
I've build the code as a command line tool. So before testing/using the code it must be made available in your path. You can for example create an alias in your .bashrc

```
alias graphRank=/path/to/the/library/graphRank.py

```

You can add the file to your bin or add the folder to your path. To test the code first go to the test folder

```
cd test_code2
```

As explained above the default values for the trainIDs and testIDs are 'testIDs.lst' and 'trainIDs.lst'. So you don't need to specify them if you keep the same file names as you currently do. Similarly the individual graphs are expected in './graphMAT' and the matlab computed kernels are expected in './kernelMAT/<test_ID>.mat'. So you don't need to specify them either as long as you keep the folder names the same. Therefore you can test the CPU/GPU version of the code with:

#### CPU version
```
graphRank --test
```

#### GPU version
```
graphRank --test --cuda
```

which should output (GPU version)

```
--------------------
- timing
--------------------

GPU - Kern : 0.111562
GPU - Mem  : 0.190918 	 (block size:8x8)
GPU - Kron : 0.081629 	 (block size:8x8)
GPU - Px   : 0.002048 	 (block size:8x8)
GPU - W0   : 0.001714 	 (block size:8x8)
CPU - K    : 0.024109

--------------------
- Accuracy
--------------------

K      :  1.57e-05  4.61e-05  0.000175  0.000491  0.00192
Kcheck :  1.57e-05  4.61e-05  0.000175  0.000491  0.00192
```

The timing part output the execution time for the main steps of the calculation. 

  * GPU - Kern : time needed to compile the cuda kernel
  * GPU - Mem  : time needed to book the memeory on the GPU

These two steps are needed only once when calculating the kernels of several pairs.

  * GPU - Kron : time needed to compute the kronecker matrix
  * GPU - Px   : time needed to compute the Px vector
  * GPU - W0   : time needed to compute the W0 matrix
  * CPU - K    : time needed to compute the kernels

The last step can only be done on CPU as it won't be much faster on GPUs. 
The code then output the values of the kernel calculated for the pair that was tested. If a valid .mat file containing the matlab precomputed kernel was found (typically ./kernelMAT/K_testID.mat), the code will also output these values for comparison.

# Kernel Tuner

The performance of the GPU code depends a lot on the number of threads and block size used. We can determine the best block size using the kernel tuner. You can tune the gpu block/grid size using the kernel tuner. Simply type:

```
graphRank --tune_kernel [--func=<func_name>]
```

If you don't specify a function name (present in cuda_kernel.c) the code will tune all the functions. For each function it should output something like:

```
Tuning function create_kron_mat from ./cuda_kernel.c
----------------------------------------
Using: GeForce GTX 1080 Ti
block_size_x=2, block_size_y=2, time=0.905830395222
block_size_x=2, block_size_y=4, time=0.545791995525
block_size_x=2, block_size_y=8, time=0.355219191313
block_size_x=2, block_size_y=16, time=0.30387840271
block_size_x=2, block_size_y=32, time=0.27014400363
block_size_x=2, block_size_y=64, time=0.259091204405
block_size_x=2, block_size_y=128, time=0.250815996528
......
best performing configuration: block_size_x=8, block_size_y=8, time=0.161958396435
```

# Run

You can run the calculation on the entire training/test set using

```
graphRank [--cuda] [--lamb=X] [--walk=X] [--outfile=name] [--gpu_block=i j k]
```

In the GPU case the code will first output the timing of the kernel compilation and GPU memory assignement. Once again these two steps are needed to be done only once.

```
GPU - Kern : 0.106779
GPU - Mem  : 0.146905
```


Then for each pair of graph present in the train/test set the code will output the following

```
7CEI_100w 4CPA
--------------------
GPU - Mem  : 0.001109    (block size:8x8)
GPU - Kron : 0.002521    (block size:8x8)
GPU - Px   : 0.001092    (block size:8x8)
GPU - W0   : 0.001091    (block size:8x8)
CPU - K    : 0.000621
--------------------
K      :  0.000245  0.000402  0.00117  0.00166  0.00445
Kcheck :  0.000245  0.000402  0.00117  0.00166  0.00445
```

As you can see if a check file (typically ./kernelMAT/K_testID.mat) is found it will also compare the values of the matlab code with the one calculated here. 


# Results

After the run the results will be dumped in a pickle file with default name kernel.pkl. You can read this file following

```python
import pickle
fname = kernel.pkl
K = pickle.load(open(fname,'rb'))
```

K is then a dictionary with the following keys:

```
K['lambda']    : lambda value used for the calculation
K['walk']      : walk length  used for the calculation
K['cuda']      : was cuda used during the calcultion (useful ?)
K['gpu_block'] : the gpu block size during the calculation (useful ?)
K[(MOL1,MOL2)] : the values of the kernel calculated for this specific pair
K[(MOL1,MOL3)] : the values of the kernel calculated for this specific pair
K[(MOL2,MOL3)] : the values of the kernel calculated for this specific pair
....
```

Using this results you can compare the python and matlab kernel values using the following script

```python
import matplotlib.pyplot as plt 
import scipy.io as spio
import pickle

# matlab kernel file
matlab = './kernelMAT/K_smalltestID.mat'

# python kernel file
python = './kernel.pkl'

# load the data
Kcheck = spio.loadmat(matlab)['K']
K = pickle.load(open(python,'rb'))

# plot the data
N = len(Kcheck)
keys = list(K.keys())[4:]
k = 0
for n1 in range(N):
  M = len(Kcheck[n1])
  for n2 in range(M):
    plt.scatter(Kcheck[n1][n2],K[keys[k]])
    k +=1
plt.show()
```
