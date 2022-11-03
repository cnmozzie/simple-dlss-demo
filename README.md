# simple-dlss-demo
A simple demo with two input color and depth arrays and output a super-resolution array

## Requirements

- An __NVIDIA GPU__; tensor cores increase performance when available.
- A __C++14__ capable compiler. The following choices are recommended and have been tested:
  - __Windows:__ Visual Studio 2019
  - __Linux:__ GCC/G++ 7.5 or higher
- [Vulkan SDK](https://vulkan.lunarg.com/) for DLSS support.

## Compilation (Windows & Linux)

Begin by cloning this repository and all its submodules using the following command:
```sh
$ git clone --recursive https://github.com/cnmozzie/simple-dlss-demo
$ cd simple-dlss-demo
```

Then, use CMake to build the project: (on Windows, this must be in a [developer command prompt](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_prompt))
```sh
instant-ngp$ cmake . -B build
instant-ngp$ cmake --build build --config RelWithDebInfo -j
```

If the build succeeds, you can now run the code via the `build/testbed_dlss` executable.
