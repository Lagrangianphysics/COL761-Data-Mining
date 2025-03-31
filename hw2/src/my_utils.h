#include <omp.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>
#include <ctime>
#include <map>
#include <iomanip>   // <-- Required for setprecision
#include <limits>    // <-- Required for numeric_limits
// #include <mutex>
#include <cmath>

using namespace std;
// mutex initGain2_mutex ;

// Define the weighted graph as an adjacency list.
// Each edge is represented as a pair: (destination, probability).
using Graph = vector<vector<pair<int, double>>>;

int INITIAL_ITR = 200 ;

// Parallelized simulation of the Independent Cascade (IC) model.
// Given a seed set (seed), returns the average number of activated nodes over
// 'iterations' simulations.

double independentCascade(const Graph &G , const vector<int> &seed , int iterations) {
    int n = G.size();
    double totalSpread = 0.0;
  
  #pragma omp parallel for reduction(+ : totalSpread)
for(int iter = 0; iter < iterations; iter++) {
    // Each thread gets its own RNG seeded with a combination of time,
    // iteration, and thread id.
    unsigned int threadSeed = static_cast<unsigned int>(
        chrono::steady_clock::now().time_since_epoch().count() + iter +
        omp_get_thread_num());
    mt19937 rng(threadSeed);

    vector<bool> activated(n,false);
    queue<int> activeQueue;
    for(int i : seed) {
    activeQueue.push(i);
    activated[i] = true ;
    }
    while(!activeQueue.empty()) {
    int current = activeQueue.front();
    activeQueue.pop();
    for(const auto &edge : G[current]) {
        int neighbor = edge.first;
        double p_edge = edge.second;
        if(!activated[neighbor]) {
        uniform_real_distribution<double> dist(0.0, 1.0);
        if(dist(rng) < p_edge) {
            activated[neighbor] = true;
            activeQueue.push(neighbor);
        }
        }
    }
    }
    totalSpread += count(activated.begin(), activated.end(), true);
}
return totalSpread / iterations;
}
  



// Structure to hold candidate node information for CELF.
struct Candidate {
    int node;
    double marginalGain;
    int lastUpdate;  // Number of seeds selected when this gain was computed.
};

// Comparator for the max-heap (priority queue).
struct CandidateCompare {
    bool operator()(const Candidate &a, const Candidate &b) {
        return a.marginalGain < b.marginalGain;  // Highest gain on top.
    }
};


int no_itr(double prev_mg) {
    double itr = 14200.0 ;
    itr -= ((double)350)*prev_mg ;
    int ans = (int)floor(itr) ;
    if(ans < 200) {
        ans = 200 ;
    } else if(ans > 10000) {
        ans = 10000 ;
    }
    return ans ;
}

// varing # MC sim exponantialy fromm 200 to 10k
vector<int> no_itr_new_arr = {200 , 326 , 531 , 867 , 1414 , 2306 , 3760 , 6132 , 10000} ;

int no_itr_new(int k , int seed_size) {
    int N = (int)(no_itr_new_arr.size()) ;
    int s = 0 ; int e = N-1 ;
    int ans = no_itr_new_arr[0] ;
    while(s <= e) {
        int mid = (s+e)/2 ;
        int siz = ((mid+1)*k) / (N+1) ;
        if(siz <= seed_size) {
            ans = no_itr_new_arr[mid] ;
            s = mid+1 ;
        } else {
            e = mid-1 ;
        }
    }
    return ans ;
}


// Optimized CELF algorithm with parallel initial gain computation and lazy
// updates.
vector<int> CELF(const Graph &G, int k, int iterations , string &outputPath , map<int,int> &new_to_org) {
    
    ofstream outFile(outputPath);
    // clear the outputfile
    outFile.close() ;

    // clock_t CELF_START = clock();
    int n = G.size();
    vector<int> selected;

    priority_queue<Candidate, vector<Candidate>, CandidateCompare> heap;

    cout << "pre computation for clef is running ..." << endl ;

    // Parallelise initial marginal gain calculation.
    vector<double> initGain(n, 0.0);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        vector<int> tempSeed ;
        tempSeed.push_back(i) ;
        initGain[i] = independentCascade(G, tempSeed, INITIAL_ITR);
    }
    for (int i = 0; i < n; i++) {
        heap.push({i, initGain[i], 0});
    }
  
    double currentSpread = 0.0;
    clock_t LAST_UPDATE = clock();
    // CELF main loop.
    int MAIN_ITR = INITIAL_ITR ;
    double prev_mg = (double)n ;
    while (selected.size() < static_cast<size_t>(k) && !heap.empty()) {

        double tt = ((double)(clock() - LAST_UPDATE) / CLOCKS_PER_SEC) ;
        if(tt > (double)(200) && MAIN_ITR > 1000) {
            MAIN_ITR = 1000 ;
        }

        Candidate top = heap.top();
        heap.pop();

        // If this candidate's marginal gain is up-to-date, select it.
        if (top.lastUpdate == static_cast<int>(selected.size())) {
            selected.push_back(top.node);
            currentSpread = independentCascade(G, selected, MAIN_ITR) ;
            
            std::ofstream file(outputPath, std::ios::app); // Open file in append mode
            file << new_to_org[top.node] << endl ;
            file.close();


            cout << "size: " << (int)(selected.size()) 
                << " node: " << new_to_org[top.node] << "\tmg: " << top.marginalGain << "\tsp: " << currentSpread 
                << "\t#_mc_itr: " << MAIN_ITR << "\tT: " << tt << endl ;
            LAST_UPDATE = clock() ;

            if(top.marginalGain > prev_mg && MAIN_ITR < 10000) {
                MAIN_ITR += 500 ;
            }

            prev_mg = top.marginalGain ;

            // MAIN_ITR = no_itr(top.marginalGain) ;
            // MAIN_ITR = no_itr_new(k , (int)(selected.size())) ;

        } else {
        // Recompute the marginal gain given the current seed set.
            vector<int> newSeed = selected;
            newSeed.push_back(top.node);
            double newSpread = independentCascade(G, newSeed, MAIN_ITR) ;
            double newGain = newSpread - currentSpread;
            top.marginalGain = newGain;
            top.lastUpdate = selected.size();
            heap.push(top);
        }
    }
    cout << currentSpread << '\n';
    return selected;
}





















// // OVERLOAD FUNCTION
// // Optimized CELF algorithm with parallel initial gain computation and lazy
// // updates.
// vector<int> CELF(const Graph &G, int k, int iterations , vector<int> initial_seed) {
//     int n = G.size();
//     vector<int> selected;

//     priority_queue<Candidate, vector<Candidate>, CandidateCompare> heap;

//     cout << "pre computation for clef is running ..." << endl ;

//     // Parallelise initial marginal gain calculation.
//     vector<double> initGain(n, 0.0);
// #pragma omp parallel for
//     for(int i = (int)(initial_seed.size()) - 1 ; i >= 0 ; i--) {
//         vector<int> tempSeed;
//         tempSeed.push_back(initial_seed[i]) ;
//         initGain[initial_seed[i]] = independentCascade(G, tempSeed, INITIAL_ITR);
//     }
//     for (int i : initial_seed) {
//         heap.push({i, initGain[i], 0});
//     }
  
//     double currentSpread = 0.0;
//     clock_t LAST_UPDATE = clock();
//     int MAIN_ITR = INITIAL_ITR + 100 ;
//     // CELF main loop.
//     while(selected.size() < static_cast<size_t>(k) && !heap.empty()) {
//         Candidate top = heap.top();
//         heap.pop();

//         // If this candidate's marginal gain is up-to-date, select it.
//         if (top.lastUpdate == static_cast<int>(selected.size())) {
//             selected.push_back(top.node);
//             currentSpread = independentCascade(G, selected, MAIN_ITR);

//             cout << "size: " << (int)(selected.size()) 
//                 << " node: " << top.node << "\tmg: " << top.marginalGain << "\tsp: " << currentSpread 
//                 << "\t#_mc_itr: " << MAIN_ITR << "\tT: " << ((double)(clock() - LAST_UPDATE) / CLOCKS_PER_SEC) << endl ;
//             LAST_UPDATE = clock() ;
            
//             MAIN_ITR = no_itr(top.marginalGain) ;

//         } else {
//         // Recompute the marginal gain given the current seed set.
//             vector<int> newSeed = selected;
//             newSeed.push_back(top.node);
//             double newSpread = independentCascade(G, newSeed, MAIN_ITR);
//             double newGain = newSpread - currentSpread;
//             top.marginalGain = newGain;
//             top.lastUpdate = selected.size();
//             heap.push(top);
//         }
//     }
//     cout << currentSpread << '\n';
//     return selected;
// }


// // CHOOSE 2 CLEF 
// // Structure to hold candidate node information for CELF.
// struct Candidate2 {
//     int node1;
//     int node2;
//     double marginalGain;
//     int lastUpdate;  // Number of seeds selected when this gain was computed.
// };

// // Comparator for the max-heap (priority queue).
// struct CandidateCompare2 {
//     bool operator()(const Candidate2 &a, const Candidate2 &b) {
//         return a.marginalGain < b.marginalGain;  // Highest gain on top.
//     }
// };

// int no_itr_2(double prev_mg) {
//     prev_mg /= 2 ;
//     double itr = 14200.0 ;
//     itr -= ((double)350)*prev_mg ;
//     int ans = (int)floor(itr) ;
//     if(ans < 200) {
//         ans = 200 ;
//     } else if(ans > 10000) {
//         ans = 10000 ;
//     }
//     return ans ;
// }
// // k should be even
// vector<int> CELF_choose2(const Graph &G, int k, int iterations , vector<int> initial_seed) {
//     int n = G.size() ;
//     vector<int> selected ;
//     vector<bool> is_seed(n, false) ;

//     // Priority queue for candidates.
//     priority_queue<Candidate2, vector<Candidate2>, CandidateCompare2> heap ;

//     cout << "pre computation for clef is running ..." << endl ;

//     // Parallelize initial marginal gain calculation.
//     map<pair<int,int>, double> initGain2 ;
// // #pragma omp parallel for
//     for(int i = (int)(initial_seed.size()) - 1; i >= 0; i--) {
//         for(int j = i - 1; j >= 0; j--) {
//             vector<int> tempSeed ;
//             tempSeed.push_back(initial_seed[i]) ;
//             tempSeed.push_back(initial_seed[j]) ;
//             double result = independentCascade(G, tempSeed, 100);
//             heap.push({initial_seed[i] , initial_seed[j] , result , 0}) ;
//             // // Protect shared access with a mutex
//             // {
//             //     lock_guard<mutex> lock(initGain2_mutex);
//             //     initGain2[{initial_seed[i], initial_seed[j]}] = result;
//             // }
//         }
//     }
//     // for(int i = (int)(initial_seed.size()) - 1; i >= 0; i--) {
//     //     for(int j = i - 1; j >= 0; j--) {
//     //         heap.push({initial_seed[i], initial_seed[j], initGain2[{initial_seed[i], initial_seed[j]}], 0}) ;
//     //     }
//     // }

//     clock_t LAST_UPDATE = clock();
//     double currentSpread = 0.0;
//     int MAIN_ITR = INITIAL_ITR + 100 ;
//     // CELF main loop.
//     while((int)(selected.size()) + 2 <= static_cast<size_t>(k) && !heap.empty()) {
//         Candidate2 top = heap.top();
//         heap.pop();

//         if(is_seed[top.node1] || is_seed[top.node2]) {
//             continue ;
//         }

//         // If this candidate's marginal gain is up-to-date, select it.
//         if (top.lastUpdate == static_cast<int>(selected.size())) {
//             selected.push_back(top.node1) ;
//             selected.push_back(top.node2) ;
//             is_seed[top.node1] = true ;
//             is_seed[top.node2] = true ;
//             currentSpread = independentCascade(G, selected, MAIN_ITR) ;


//             cout << "size: " << (int)(selected.size()) 
//                 << " node: " << top.node1 << ", " << top.node2 << "\t\tmg: " << top.marginalGain << "\tsp: " << currentSpread 
//                 << "\t#_mc_itr: " << MAIN_ITR << "\tT: " << ((double)(clock() - LAST_UPDATE) / CLOCKS_PER_SEC) << endl ;
//             LAST_UPDATE = clock() ;
            
//             MAIN_ITR = no_itr_2(top.marginalGain) ;

//         } else {
//             // Recompute the marginal gain given the current seed set.
//             vector<int> newSeed = selected ;
//             newSeed.push_back(top.node1) ;
//             newSeed.push_back(top.node2) ;
//             double newSpread = independentCascade(G, newSeed, MAIN_ITR);
//             double newGain = newSpread - currentSpread;
//             top.marginalGain = newGain;
//             top.lastUpdate = selected.size();
//             heap.push(top);
//         }
//     }
//     return selected;
// }




// Function to load the weighted graph from a file.
// The file should have lines in the format: u v p
Graph loadGraph(const string &filename, int &numNodes) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    int u, v;
    double p;
    int maxNode = -1;
    vector<tuple<int, int, double>> edges;
    string line;
    while(getline(infile, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        if (!(iss >> u >> v >> p)) continue;
        if (u != v) {
            edges.push_back(make_tuple(u, v, p));
            maxNode = max(maxNode, max(u, v));
        }
    }
    infile.close();
    numNodes = maxNode + 1;
    Graph G(numNodes);
    for (const auto &edge : edges) {
        int src, dst;
        double prob;
        tie(src, dst, prob) = edge;
        G[src].push_back({dst, prob});
    }
    return G;
}


map<int,int> convert_to_zero_index(string graphPath , string Index_graph_path) {
    map<int,int> org_to_new ;
    map<int,int> new_to_org ;

    map<int , vector<int>> adjList ;
    map<pair<int , int> , double> prob_map ;

    ifstream file(graphPath) ;
    string line ;
    int nnodes = 0 , nedges = 0 ;
    int src = -1 , dst = -1 ;
    double prob = 0 ;

    while(getline(file , line)) {
        istringstream iss(line) ;

        if(line[0] == '#') {
            iss.ignore(1) ;
            iss >> nnodes ;
            iss >> nedges ;
        } else {
            if(iss >> src >> dst >> prob) {
                adjList[src].push_back(dst) ;
                prob_map[{src , dst}] = prob ;
            }
        }
    }
    file.close() ;

    int Nnodes = adjList.size() ;

    auto it = adjList.begin() ;
    for(int i = 0 ; i < Nnodes ; i++){
        long node = (*it).first ;
        it++ ;
        org_to_new[node] = i ;
        new_to_org[i] = node ;
    }

    ofstream Index_graph(Index_graph_path) ;
    Index_graph << fixed << setprecision(numeric_limits<double>::max_digits10);  // Ensuring max precision
    for(auto pr : prob_map){
        pair<long , long> edge = pr.first ;
        src = org_to_new[edge.first] ;
        dst = org_to_new[edge.second] ;
        double p = pr.second ;
        Index_graph << src << " " << dst << " " << p << endl ;
    }
    Index_graph.close();


    return new_to_org ;
}