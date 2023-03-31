CC=g++
CFLAGS=-Wall -Wextra -Werror -std=c++11 -g

all: main

main: main.o braid.o siphash.o
	$(CC) $(CFLAGS) -o main main.o braid.o siphash.o

braid.o: braid.cc braid.h main.h siphash.h
	$(CC) $(CFLAGS) -c braid.cc
main.o: main.cc braid.h main.h
	$(CC) $(CFLAGS) -c main.cc
siphash.o: siphash.cc siphash.h
	$(CC) $(CFLAGS) -c siphash.cc

clean::
	rm -f *.o main
