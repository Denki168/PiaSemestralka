all: mesh

CPP = g++
CPPFLAGS = -std=c++11 -fopenmp -O2 #-DDEBUG

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

mesh: main.o Mesh.o output.o Vector2D.o Compressible.o init.o BC.o MeshGmsh.o
	$(CPP) $(CPPFLAGS) -o mesh main.o Mesh.o output.o Vector2D.o Compressible.o init.o BC.o MeshGmsh.o

clean:
	rm -f *.o mesh
	
