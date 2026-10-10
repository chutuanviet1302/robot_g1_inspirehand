import os
import sys

if sys.platform == "win32":
    # Laptops with NVIDIA Optimus give OpenGL processes the integrated GPU by default: MuJoCo rendered a 256x256
    # camera image in 250-380 ms on the Intel iGPU and in 1.5-13 ms on the GTX 1650. This asks the Optimus shim
    # for the discrete GPU, for this process and the workers it spawns (no system setting is changed).
    os.environ.setdefault("SHIM_MCCOMPAT", "0x800000001")
