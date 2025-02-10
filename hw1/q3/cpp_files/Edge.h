#ifndef EDGE_H
#define EDGE_H

#include <iostream>
using namespace std ;

struct Edge {
    int u_id;
    int v_id;
    int label;
    Edge() {}
    Edge(int u_id, int v_id, int label) : u_id(u_id), v_id(v_id), label(label) {}
};

#endif