# Esp32AI

Simple C++17 Neural Network (NN), Convolutional Neural Network (CNN) and Deep Learning for Esp32 on IDF from scratch.

## Intro

After Espressif's ESP32-S3 and devkits derivates (such as ESP-EYE), I'd like to write an easy library in C++17 as alternative to ESP-DL and ESP-NN, working with IDF version 5.0. 

This library could also be used in Windows or Linux o.s. for testing and fun.

Style is always my own: simple code, well formatted and widely commented. Also, in order to refresh my skills on AI I will attach my remarks: this will be a library and tutorial with examples just for my fun and free time.

Project is made of a library as an IDF Component (for a tutorial) in order to use this library as source or, if preferred, as static library (.a file). Instructions on using library can be found [here](https://github.com/briand-hub/LibEsp32IDF/blob/main/README.md#use-as-source-easier) because of identical set-up (same way of using library, very well detailed!).

Project has also a running main.cpp file with a sample project containing also library testing and performance. More details in the tests and project examples section.

### Why from scratch? Why not use a famous, ready, tested library?

Because I like to do things in my way, with my style. Falling into an error, find a solution and do better next time. Because technology is my passion, my life, my future. Because I like learn-by-doing.

Because I like to learn everyday and teach to others what I have learned.

*Simply... because I love it*

<a href="https://www.buymeacoffee.com/briandhub"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=briandhub&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" /></a>

## AI TUTORIAL

**Please, take a look at the [Tutorial](AI Tutorial.md), is fun and explains what I did in this project. Very basic and for beginners, made of my own remarks and notes collected in years.**

## Coming soon



## Library contents

Unique and easy to use header file for the entire library. Just  ``#include "BriandAI.hxx"`` and you are ready to go with your C++ software.

### Library structure

All is under *Briand* namespace. Project folders and files:

```
+ components
|--+ briand_ai
|   |--+ include
    |  |-- BriandAI.hxx          Unique header to include in project
    |  |-- BriandInclude.hxx     Unique header to be included inside library files with non-esp platform porting 
    |  |-- BriandNN.hxx          Neural Network library header
    |  |-- BriandCNN.hxx         Convolutional Neural Network library header
    |  |-- BriandMath.hxx        Math library (functions needed) header
    |  |-- BriandImage.hxx       Image library header
    |
    |-- BriandNN.cpp             Sources
    |-- BriandCNN.cpp
    |-- BriandMath.cpp
    |-- BriandImage.cpp
    |-- BriandPorting.cpp
    |
    |-- CMakeLists.txt           Library build file
```

## Example projects and tests

## Using on ESP32

You can use project under any ESP32 platform with enough power and SPIRAM. I included the [sdkconfig](sdkconfig) file however you can use your own. Also check before building with menuconfig otherwise some errors may occour. I tested the setup with ESP32-WROVER and ESP32-S3.

## Using under Linux

It's enough easy! I have created a header that can do the trick (*components/briand_ai/BriandPorting.hxx*) by redefining the needed ESP-IDF functions used in the project with the linux or windows style. Of course, there are the basic ones and if you will use more on your project then you have to add them to this file.

You can see [Makefile](/platform_porting/Makefile) and adjust what you need.

In order to compile run command ``make`` and library will be compiled under windows or linux. Then main.cpp will be compiled too and executable will be created.

Tested under linux with *g++ (Debian 8.3.0-6) 8.3.0* 

## Using under Windows

Library *could* be used under windows too with same makefile. However with *g++.exe (MinGW.org GCC-6.3.0-1) 6.3.0* is not compiling because pthread library is not available on Windows. I will try to do further tests to fnid a solution.

## Library performances

### Base functions (activation, random ...)

Tested on ESP32-S3 WROOM (by Freenove kit)

```


```

### To-do list

 - [x] Start project and organize
 - [ ] Include tutorial on development and my notes/remarks about AI
 - [ ] Linux/Windows compatibility with porting library
 - [x] Decide sample project for starting
 - [ ] NN basics
 - [ ] NN tests 
 - [ ] Deep learning basics
 - [ ] Deep learning tests
 - [ ] CNN basics
 - [ ] CNN tests
 - [ ] Test memory leaks with **valgrind**
 - [ ] Library performance tests on ESP32-S3 and ESP32 WROVER (SPI-RAM devices)
 - [ ] Example project 1: OR port with NN
 - [ ] Example project 2: sum two numbers
 - [ ] Example project 3: color recognition/classifier (supervised)
 - [ ] Example project 4: color recognition/classifier (unsupervised)
 - [ ] Example project 5: human face detection (single)
 - [ ] Example project 6: human face features detection (single)
 - [ ] Example project 7: human face detection (multiple)
 - [ ] Example project 8: human face recognition (single)
 - [ ] Example project 9: human face recognition (multiple)
 - [ ] Example project 10: if all working, separate project for my idea (upcoming maybe!)
