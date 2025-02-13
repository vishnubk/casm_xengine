# Compiler
CC = gcc

# Compiler flags
CFLAGS = -g -Wall -O2

# PSRDADA library flags
PSRDADA_INCLUDE = -I/usr/local/include
PSRDADA_LIB = -L/usr/local/lib -lpsrdada

# Other libraries
LIBS = -lm -lpthread

# Targets
all: dsaX_capture dumpfil

dsaX_capture: dsaX_capture.c
	$(CC) $(CFLAGS) $(PSRDADA_INCLUDE) -o $@ $< $(PSRDADA_LIB) $(LIBS)

dumpfil: dumpfil.c
	$(CC) $(CFLAGS) $(PSRDADA_INCLUDE) -o $@ $< $(PSRDADA_LIB) $(LIBS)

clean:
	rm -f dsaX_capture dumpfil *.o

.PHONY: all clean
