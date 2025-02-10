#ifndef NODE_H
#define NODE_H

#include <iostream>
#include "Vertex_invariant.h"
using namespace std ;

class Node {
public:
    int id ;
    Vertex_invariant invariant ;
    Node() {}
    Node(int _id_, int _label_) {
        id = _id_ ;
        invariant = Vertex_invariant(_label_ , 0) ;
    }
    inline int label() const {
        return invariant.label ;
    }
    inline int degree() const {
        return invariant.degree ;
    }
};

#endif