import os

path = os.path.join( os.path.dirname(__file__), "tutorial" )
execute  = "cp -RL " + path + " "
execute += "vem_tutorial"
status = os.system(execute)
if status != 0: raise RuntimeError(status)

# remove files generated from CMake
execute  = "rm -rf "
execute += ' '.join("vem_tutorial/{}".format(f) for f in
               ["cmake_install.cmake",
                "CTestTestfile.cmake",
                "Makefile",
                "CMakeFiles",
                "src_dir"
               ])
status = os.system(execute)
if status != 0: raise RuntimeError(status)

print("##################################################################")
print("## Example scripts based on the paper ... is now located in the 'vem_tutorial' folder.")
try:
    import matplotlib
except ImportError:
    print("## Note: the examples requires the installation of 'matplotlib'.")
print("##################################################################")
