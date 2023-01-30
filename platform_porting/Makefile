# Variables to control Makefile operation

CC = g++
CFLAGS = -fpermissive -pthread -std=gnu++17 -g

MAIN_SRCPATH = ../main/
MAIN_INCLUDEPATH = ../components/briand_ai/include/
MAIN_OUTNAME = main

LIB_SRCPATH = ../components/briand_ai/
LIB_INCLUDEPATH = ../components/briand_ai/include/
LIB_OUTNAME = briand_ai

all:

# Build library first
# -c option means no linking
	
	$(CC) -c $(LIB_SRCPATH)*.cpp -I$(LIB_INCLUDEPATH) $(CFLAGS) 

# ar r makes archive of all .o objects (ru option mean substituite in existing archive if have same name)

	ar r lib$(LIB_OUTNAME).a *.o

# ranlib will update archive symbol map

	ranlib lib$(LIB_OUTNAME).a

# Build MAIN with option -L. (tells search this folder for libraries too) and -l briand_ai to link the produced output library.

	$(CC) -o $(MAIN_OUTNAME) $(MAIN_SRCPATH)*.cpp $(CFLAGS) -I$(MAIN_INCLUDEPATH) -L. -l$(LIB_OUTNAME)

# Command line samples
# g++ -c ../components/briand_ai/*.cpp -pthread -I../components/briand_ai/include/ -std=gnu++17 -fpermissive
# ar r libbriand_ai.a *.o
# ranlib libbriand_ai.a
# g++ -o main ../main/main.cpp -I../components/briand_ai/include/ -std=gnu++17 -fpermissive -l briand_ai -L.
