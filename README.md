Instructions to Run programs

Non-Tensor Programs
===================
To run the Non-Tensor Programs run Make in the home directory of the repo.
Then CD into the Scripts folder and run ./lauch.sh this script will run each of these 
programs and output it to its results to its own respective text file.

The input sizes for the Matrix Multiply Program:
- 5000 x 5000
- 1000 x 10000
- 15000 x 15000
- 20000 x 20000
- 25000 x 25000

The input sizes for the Matrix Vector Multiply Program:
- 10000 x 10000
- 20000 x 20000
- 30000 x 30000
- 40000 x 40000

The Nueral Network is untrained and uses Static parameters that can be changed, these variables are in the runDatatype function of the nn.cu program 
and are as follows:
- n_inputs       // Number of input channels
- n_filters     // Number of filters
- height       // Input height
- width        // Input width
- filter_height  // Filter height
- filter_width   // Filter width

-------------------

Tensor Program
===================

To run the Tensor Core program use the same Makefile inside the home directory of the respository, 
then CD into the scripts folder and run ./launch.sh this script will run each of these 
programs and output it to its results to its own respective text file.

There are 3 variables that can be changed inside the file to accomodate different matrix sizes these variables are the M_TILES, N_TILES, K_TILES values. When changed these values must remain a number that is a multiple of 16, otherwise the fragments in the program will not compile correctly.

-------------------

Deprecated Programs
===================

**This code is outdated but still works the more updated files are the ones listed above (not included in the automation)**

To run these programs CD into the Deprecated Code Directory then run Make.

To run the reduction program:

srun --gres=gpu ./reduction

To run the original Non-tensor Matrix Multiply with FP16:

srun --gres=gpu ./MM16

To run the original Non-tensor Matrix Multiply with FP32:

srun --gres=gpu ./MM32

-------------------

