#ifndef INVARIANT
#define INVARIANT

#include <iostream>
using namespace std ;

class Vertex_invariant{
public:
    int label = -1 ;
    int degree = 0 ;
    Vertex_invariant() : label(-1) , degree(0) {}
    Vertex_invariant(int _label_ , int _degree_) {
        label = _label_ ;
        degree = _degree_ ;
    }

    bool operator == (const Vertex_invariant& other_invariant) const {
        return (label == other_invariant.label) && (degree == other_invariant.degree) ;
    }

    bool operator != (const Vertex_invariant& other_invariant) const {
        return !(*this == other_invariant);
    }

    bool operator < (const Vertex_invariant& other_invariant) const {
        if(degree == other_invariant.degree) {
            return (label < other_invariant.label) ;
        }
        return (degree < other_invariant.degree) ;
    }

    bool operator > (const Vertex_invariant& other_invariant) const {
        if(degree == other_invariant.degree) {
            return (label > other_invariant.label) ;
        }
        return (degree > other_invariant.degree) ;
    }
    
    bool operator <= (const Vertex_invariant& other_invariant) const {
        return !(*this > other_invariant) ;
    }

    bool operator >= (const Vertex_invariant& other_invariant) const {
        return !(*this < other_invariant) ;
    }

} ;

#endif