#include "MeshGmsh.h"

void MeshGmsh::find_msh_section(std::ifstream& f,const std::string& name) {
  std::string line;
 
  while (line != name) {
//    getline(f, line);
      f >> line;
    //if(name == line)  cout << line<<"0"<<endl;
    //if(line == "$MeshFormat")  cout << line<<"1"<<endl;
    if (f.eof()) {
      std::cerr << "Not found " << name << " !" << std::endl;
      exit(1);
    }
  };
};

MeshGmsh::MeshGmsh(const std::string& name) {
  int itmp, itmp2;
  int dim, last, next;
  double x, y, z;
  int first, second, third, fourth;
  int s;
  int type_elem, nb_tags;
  int* num;

  std::cout << "Reading grid from: " << name << std::endl;

  std::ifstream f;
  try {
      f.open(name.c_str(),std::ifstream::in);
  }
  catch (const std::ifstream::failure& e) {
      //if (f == NULL) {
      std::cout << "File \"" << name << "\" doesn't exist or cannot be opened!" << std::endl;
      exit(1);
  }
  
  find_msh_section(f, "$Nodes");
  int nn;
  f >> nn; // number of nodes;
  for (int i=0; i<nn; i++) {
    f >> itmp2>> x >> y >> z;

    node.push_back(Point(x,y));
  }
  std::cout << "Reading "<< nn << " nodes" << std::endl;

  find_msh_section(f, "$Elements");
  int n_elem;
  f >> n_elem;
  
  std::vector<Edge> helper_edges; // Node n1 id, node n2 id, edge location number

  for (int i=0; i<n_elem; i++) {
    f >> itmp>> type_elem    >> nb_tags;
    num = new int[nb_tags];

    for(int itag=0;itag<nb_tags;itag++)	f >> num[itag] ;

    switch(type_elem)
      {
      case 1:
		f>> last >> next;
		helper_edges.push_back(Edge(last-1,next-1,0.0,*this));
		helper_edges.back().location = num[0]; // ID of the boundary location
		break;

      case 2:
		f >> first >> second >> third;
		cell.push_back(Polygon({first-1,second-1,third-1},*this));
		break;

      default: std::cout << "unauthorized type_elem "<< std::endl;exit(1);
      }
    delete [] num;
  }
  nc = cell.size();
  std::cout << "Reading " << cell.size() << " cells" << std::endl;
  
  initPointCellNeighbors();
  generateEdges();
  initLeftRight();
  addGhostCells();
    
  // Mark boundary edges with their location numbers
  for (auto& e : edge) {
  	if (e.boundary) {
  	  for (auto h : helper_edges) {
   	    if ((e.n1 == h.n1 && e.n2 == h.n2) || (e.n1 == h.n2 && e.n2 == h.n1)) {
   	  	  e.location = h.location;
   	    }
   	  }
   	}
  }
  
  
};