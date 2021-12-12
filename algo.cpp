#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <iomanip>
#include <set>

#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

// begin trivial helper stuff
ostream& dbg = cerr;

void fail (const string &s) {
    cout << "FAIL: " << s << endl;
    dbg << "FAIL: " << s << endl;
    exit(1);
}

void warn (const string &s) {
    dbg << "WARNING: " << s << endl;
}

#define DBG(vari) cerr<<"["<<__LINE__<<"] "<<#vari<<" = "<<(vari)<<endl;

template <typename T>
ostream& operator << (ostream &s, const vector<T> &v) {
    for (const T &x : v) {
        s << x << " ";
    }
    return s;
}

template <typename T>
string to_string (const vector<T> &v) {
    stringstream ss;
    ss << v;
    return ss.str();
}

template <typename T>
void append (vector<T> &v, const vector<T> &w) {
    v.insert(v.end(), w.begin(), w.end());
}

template <typename T>
inline void minify (T &x, const T &y) {
    x = min(x,y);
}

int ceildiv (int x, int y) {
    assert(y > 0);
    return (x + y - 1) / y;
}

constexpr double INFTY = 1e30;

vector<int> vectorOfSetBits (const vector<bool> &v) {
    vector<int> res;
    for (int i = 0; i < v.size(); ++i) {
        if (v[i]) {
            res.push_back(i);
        }
    }
    return res;
}

string formatMemoryUsage (const double M) {
    stringstream ss;
    ss << setprecision(1) << fixed << (M / (1 << 30)) << " GB";
    return ss.str();
}

string formatTPS (double our) {
    stringstream ss;
    int prec = 1;
    // make it so that there are 3 nonzero digits displayed
    double powerOur = our * 10;
    while (powerOur < 1 && prec < 3) {
        prec++;
        powerOur *= 10;
    }
    ss << setprecision(prec+2) << fixed << our << " s";
    return ss.str();
}

string formatTPSIncrease (double our, double worse) {
    if (worse > INFTY/2) {
        return "OOM";
    }
    stringstream ss;
    double percent = (worse / our - 1) * 100;
    if (percent < 100) {
        ss << setprecision(2) << fixed << percent << "\\%";
    } else {
        double times = worse / our;
        ss << setprecision(2) << fixed << times << "x";
    }
    return ss.str();
}

string formatRuntime (double our) {
    stringstream ss;
    ss << setprecision(1) << fixed << our << " s";
    return ss.str();
}

double average(const vector<double> &numbers) {
   if (numbers.empty()) {
      fail("trying to take average of an empty vector");
   }
   return accumulate(numbers.begin(), numbers.end(), 0.0) / static_cast<double>(numbers.size());
}

double sampleStddev(const vector<double> &numbers) {
   if (numbers.empty()) {
      fail("trying to take stddev of an empty vector");
   }
   if (numbers.size() == 1) {
      fail("trying to take sample stddev of a single number");
   }
   const double avg = average(numbers);
   double sum_of_squares = 0.0;
   for (double number : numbers) {
      sum_of_squares += pow(number - avg, 2);
   }
   return sqrt(sum_of_squares / (static_cast<double>(numbers.size()) - 1));
}
// end trivial helper stuff


constexpr int IDEALS_LIMIT = 40'000;
constexpr int IDEALS_EXPLORATION_LIMIT = 200'000;
constexpr int DEVICES_LIMIT = 10'000; // some loose upper bound on number of devices there can be in any reasonable input
bool DATA_PARALLELISM_ALLOWED = true;
bool TENSOR_PARALLELISM_ALLOWED = true;
bool ACTIVATION_RECOMPUTATION_ALLOWED = true;
bool ACTIVATION_RECOMPUTATION_FORCED = false;
bool ACTIVATION_RECOMPUTATION_ALL_LAYERS_OR_NONE = false;
constexpr bool KNAPSACK_FAST_HEURISTIC = true;
bool OUTPUT_KNAPSACK_INSTANCES_FOR_INSPECTION = false;
bool DEBUG_DATA_PARALLEL_COSTS = false;
constexpr bool FASTER_DP_IMPLEMENTATION = true;


struct TMPC {
    // known from the context:
    // - node id (v)
    // - number of devices (k)
    double timePerSample; // p
    unordered_map<int,double> syncTimeFw; // syncTimeFw[u] = sfw(u, this node)
    unordered_map<int,double> syncTimeBw; // syncTimeBw[w] = sbw(this node, w)
    double parameterSize; // w (only used to compute data-parallel resync costs)
    double memoryUsageA, memoryUsageB; // usage is A*y + B, y being ceil(suffix-sum-of-dp-degrees / d)
    string id;
};


void from_json (const json &j, TMPC &t) {
    j.at("timePerSample").get_to(t.timePerSample);
    map<string,double> syncTimeFw, syncTimeBw;
    j.at("syncTimeFw").get_to(syncTimeFw);
    for (const pair<string,double> &p : syncTimeFw) {
        // TODO verify that p.first is a number
        t.syncTimeFw[stoi(p.first)] = p.second;
    }
    j.at("syncTimeBw").get_to(syncTimeBw);
    for (const pair<string,double> &p : syncTimeBw) {
        // TODO verify that p.first is a number
        t.syncTimeBw[stoi(p.first)] = p.second;
    }
    j.at("parameterSize").get_to(t.parameterSize);
    j.at("memoryUsageA").get_to(t.memoryUsageA);
    j.at("memoryUsageB").get_to(t.memoryUsageB);
    j.at("id").get_to(t.id);
}


struct Node {
    int id; // v
    unordered_map<int,vector<TMPC>> TMPCs; // TMPCs[k] = vector of TMPCs for number k of devices
    string name; // just for debugging etc.
};


void from_json (const json &j, Node &n) {
    j.at("id").get_to(n.id);
    map<string,vector<TMPC>> TMPCs;
    j.at("TMPCs").get_to(TMPCs);
    for (const pair<string,vector<TMPC>> &p : TMPCs) {
        // TODO verify that p.first is a number
        n.TMPCs[stoi(p.first)] = p.second;
    }
    if (j.count("name")) {
        j.at("name").get_to(n.name);
    }
}


struct Edge {
    int sourceId; // u
    int destId;  // v
    double communicationCost; // c(u,v), in bytes
};


void from_json (const json &j, Edge &e) {
    j.at("sourceId").get_to(e.sourceId);
    j.at("destId").get_to(e.destId);
    j.at("communicationCost").get_to(e.communicationCost);
}


struct Instance {
    double maxMemoryPerDevice;
    int maxDevices;
    double bandwidth; // in bytes per second
    int mbsInBatch;
    vector<Node> nodes;
    vector<Edge> edges;

    // filled with renumber()
    unordered_map<int,int> newNumber;
    vector<int> oldNumber;

    void linearize();
    void checkInputCorrectness() const;
    bool isDAG() const;
    void renumber();

    // for our experiments
    void insertTransformerLayer();
    vector<int> getTransformerIds() const;
    vector<int> getNodeIdsToposorted() const;
    int getMaxTensorParallelDegree() const;
};


void from_json (const json &j, Instance &ii) {
    j.at("maxMemoryPerDevice").get_to(ii.maxMemoryPerDevice);
    j.at("maxDevices").get_to(ii.maxDevices);
    j.at("bandwidth").get_to(ii.bandwidth);
    j.at("mbsInBatch").get_to(ii.mbsInBatch);
    j.at("nodes").get_to(ii.nodes);
    j.at("edges").get_to(ii.edges);

    ii.checkInputCorrectness();
    ii.renumber();
    ii.checkInputCorrectness();
}


void Instance::linearize () {
    fail("not needed for this work, to be implemented when needed");
    // remember, there should be no parallel edges even after adding the path
}


void Instance::checkInputCorrectness() const {
    if (maxDevices < 1 || maxDevices > DEVICES_LIMIT) {
        fail("wrong number of devices");
    }
    if (bandwidth < 1e-9) {
        fail("wrong bandwidth");
    }
    if (maxMemoryPerDevice < 1e-9) {
        fail("wrong maxMemoryPerDevice");
    }
    if (mbsInBatch < 1) {
        fail("wrong mbsInBatch");
    }
    if (nodes.empty()) {
        fail("no nodes in input");
    }
    set<pair<int,int>> edgesAsPairs;
    unordered_map<int,unordered_set<int>> incomingEdges, outgoingEdges;
    for (const Node &n : nodes) {
        incomingEdges[n.id] = {};
        outgoingEdges[n.id] = {};
    }
    if (edges.empty()) {
        fail("no edges in input");
    }
    for (const Edge &e : edges) {
        if (e.communicationCost < 0) {
            fail("communication cost of edge is negative"); // zero is fine
        }
        if (edgesAsPairs.insert(make_pair(e.sourceId, e.destId)).second == false) {
            fail("duplicated edge");
        }
        if (!incomingEdges.count(e.destId) || !outgoingEdges.count(e.sourceId)) {
            fail("edge endpoint is not in nodes");
        }
        incomingEdges[e.destId].insert(e.sourceId);
        outgoingEdges[e.sourceId].insert(e.destId);
    }

    if (!isDAG()) {
        fail("instance is not acyclic");
    }

    set<int> nodeIds;
    for (const Node &n : nodes) {
        if (nodeIds.insert(n.id).second == false) {
            fail("duplicate node ID");
        }
        // check TMPCs
        for (const auto &it : n.TMPCs) {
            if (it.first < 1 || it.first > DEVICES_LIMIT) {
                fail("key " + to_string(it.first) + " doesn't look like a number of devices");
            }
            set<string> TMPCIds;
            for (const TMPC &tmpc : it.second) {
                if (tmpc.id.empty()) {
                    fail("TMPC ID empty");
                }
                if (TMPCIds.insert(tmpc.id).second == false) {
                    fail("TMPC IDs are not distinct within one node and number of devices");
                }
                // check each TMPC
                if (tmpc.timePerSample < 0) {
                    fail("wrong timePerSample");
                }
                if (tmpc.parameterSize < 0) {
                    fail("wrong parameterSize");
                }
                if (tmpc.memoryUsageA < 0 || tmpc.memoryUsageB < 0) {
                    fail("wrong memoryUsage");
                }
                // syncTimes should contain only real edges
                for (const pair<int,double> &p : tmpc.syncTimeFw) {
                    if (p.second < 0) {
                        fail("wrong syncTime");
                    }
                    if (!incomingEdges[n.id].count(p.first)) {
                        dbg << n.id << "<-" << p.first << endl;
                        fail("edge present in syncTimes but not in edges");
                    }
                }
                for (const pair<int,double> &p : tmpc.syncTimeBw) {
                    if (p.second < 0) {
                        fail("wrong syncTime");
                    }
                    if (!outgoingEdges[n.id].count(p.first)) {
                        dbg << n.id << "->" << p.first << endl;
                        fail("edge present in syncTimes but not in edges");
                    }
                }
                // and they should contain all edges
                for (int u : incomingEdges[n.id]) {
                    if (!tmpc.syncTimeFw.count(u)) {
                        dbg << "incoming " << n.id << " <- " << u << endl;
                        fail("edge missing from syncTimes");
                    }
                }
                for (int w : outgoingEdges[n.id]) {
                    if (!tmpc.syncTimeBw.count(w)) {
                        dbg << "outgoing " << n.id << " -> " << w << endl;
                        fail("edge missing from syncTimes");
                    }
                }
            }
            if (KNAPSACK_FAST_HEURISTIC) {
                if (it.second.size() > 2) {
                    fail("can't run the fast heuristic with > 2 TMPCs");
                }
            }
        }
        if (!n.TMPCs.count(1) || n.TMPCs.at(1).empty()) {
            warn("no TMPC given for executing node " + to_string(n.id) + " w/o tensor parallelism?");
        }

    }
}


bool Instance::isDAG() const {
    unordered_map<int,int> indegree;
    unordered_map<int,vector<int>> outgoingEdges;
    for (const Edge &e : edges) {
        ++indegree[e.destId];
        outgoingEdges[e.sourceId].push_back(e.destId);
    }
    vector<int> deg0vertices;
    for (const Node &n : nodes) {
        if (indegree[n.id] == 0) {
            deg0vertices.push_back(n.id);
        }
    }
    int processed_vertices = 0;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back();
        deg0vertices.pop_back();
        ++processed_vertices;
        for (int w : outgoingEdges[v]) {
            --indegree[w];
            if (indegree[w] == 0) {
                deg0vertices.push_back(w);
            }
        }
    }
    return processed_vertices == nodes.size();
}


void Instance::renumber () {
    // renumber nodes as 0,1,2,...
    assert(oldNumber.empty()); // oldNumber.size() used as next id
    // build oldNumber and newNumber
    for (Node &n : nodes) {
        newNumber[n.id] = oldNumber.size();
        oldNumber.push_back(n.id);
    }
    // now replace old ids with new ids everywhere
    for (Node &n : nodes) {
        n.id = newNumber[n.id];
        for (auto &it : n.TMPCs) {
            for (auto &tmpc : it.second) {
                unordered_map<int,double> newSyncTimeFw, newSyncTimeBw;
                for (const auto &p : tmpc.syncTimeFw) {
                    newSyncTimeFw[newNumber[p.first]] = p.second;
                }
                for (const auto &p : tmpc.syncTimeBw) {
                    newSyncTimeBw[newNumber[p.first]] = p.second;
                }
                tmpc.syncTimeFw = move(newSyncTimeFw);
                tmpc.syncTimeBw = move(newSyncTimeBw);
            }
        }
    }
    for (Edge &e : edges) {
        e.sourceId = newNumber[e.sourceId];
        e.destId = newNumber[e.destId];
    }
}


void Instance::insertTransformerLayer () {
    // make a larger DNN for our experiments

    // find a ParallelTransformerLayer
    unordered_set<int> idsOfTransformerLayers;
    for (const Node &n : nodes) {
        if (n.name == "ParallelTransformerLayer") {
            idsOfTransformerLayers.insert(n.id);
        }
    }
    unordered_map<int,int> prevTransformer, nextTransformer;
    for (const Edge &e : edges) {
        if (idsOfTransformerLayers.count(e.sourceId) && idsOfTransformerLayers.count(e.destId)) {
            prevTransformer[e.destId] = e.sourceId;
            nextTransformer[e.sourceId] = e.destId;
        }
    }
    int v = -1;
    for (int w : idsOfTransformerLayers) {
        if (prevTransformer.count(w) && nextTransformer.count(w)) {
            // found
            v = w;
            break;
        }
    }
    if (v == -1) {
        fail("no transformer with transformer precedessor and successor found");
    }

    // replicate v
    static int freshNumber = 123456789;
    ++freshNumber;
    int nu = oldNumber.size();
    oldNumber.push_back(freshNumber);
    newNumber[freshNumber] = nu;

    const int s = nextTransformer[v];
    assert(nodes[v].id == v);
    assert(nodes[s].id == s);
    nodes.emplace_back();
    nodes.back().id = nu;
    nodes.back().name = "ParallelTransformerLayer";

    // redirect edge v->s to be v->nu
    // and add edge nu->s
    double communicationCost = 1e30;
    for (Edge &e : edges) {
        if (e.sourceId == v && e.destId == s) {
            e.destId = nu;
            communicationCost = e.communicationCost;
        }
    }
    assert(communicationCost < 1e29);
    edges.emplace_back();
    edges.back().sourceId = nu;
    edges.back().destId = s;
    edges.back().communicationCost = communicationCost;

    for (auto &it : nodes[v].TMPCs) {
        for (auto &tmpc : it.second) {
            const double sbw = tmpc.syncTimeBw.at(s);
            tmpc.syncTimeBw.erase(s);
            tmpc.syncTimeBw.emplace(nu, sbw);
        }
    }
    for (auto &it : nodes[s].TMPCs) {
        for (auto &tmpc : it.second) {
            const double sfw = tmpc.syncTimeFw.at(v);
            tmpc.syncTimeFw.erase(v);
            tmpc.syncTimeFw.emplace(nu, sfw);
        }
    }
    // create TMPCs for nu
    nodes.back().TMPCs = nodes[v].TMPCs;
    for (auto &it : nodes.back().TMPCs) {
        for (auto &tmpc : it.second) {
            // leave all stuff as it was, just rewrite syncTimeFw and syncTimeBw
            const double sfw = tmpc.syncTimeFw.at(prevTransformer[v]);
            const double sbw = tmpc.syncTimeBw.at(nu);
            tmpc.syncTimeFw = {{v, sfw}};
            tmpc.syncTimeBw = {{s, sbw}};
        }
    }

    // TODO: ideally, here we should also:
    // for each other edge a->v, add a->nu
    // for each other edge v->b, add nu->b
    // (but, in the BERT model input, the former are irrelevant and latter are absent)

    checkInputCorrectness();
}


vector<int> Instance::getTransformerIds() const {
    unordered_set<int> tranformerIdsUnordered;
    for (const Node &n : nodes) {
        if (n.name == "ParallelTransformerLayer") {
            tranformerIdsUnordered.insert(n.id);
        }
    }

    // return these in topological order
    unordered_map<int,int> indegree;
    unordered_map<int,vector<int>> outgoingEdges;
    for (const Edge &e : edges) {
        ++indegree[e.destId];
        outgoingEdges[e.sourceId].push_back(e.destId);
    }
    vector<int> deg0vertices;
    for (const Node &n : nodes) {
        if (indegree[n.id] == 0) {
            deg0vertices.push_back(n.id);
        }
    }
    vector<int> transformerIdsOrdered;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back();
        deg0vertices.pop_back();
        if (tranformerIdsUnordered.count(v)) {
            transformerIdsOrdered.push_back(v);
        }
        for (int w : outgoingEdges[v]) {
            --indegree[w];
            if (indegree[w] == 0) {
                deg0vertices.push_back(w);
            }
        }
    }
    return transformerIdsOrdered;
}


vector<int> Instance::getNodeIdsToposorted() const {
    unordered_map<int,int> indegree;
    unordered_map<int,vector<int>> outgoingEdges;
    for (const Edge &e : edges) {
        ++indegree[e.destId];
        outgoingEdges[e.sourceId].push_back(e.destId);
    }
    vector<int> deg0vertices;
    for (const Node &n : nodes) {
        if (indegree[n.id] == 0) {
            deg0vertices.push_back(n.id);
        }
    }
    vector<int> nodeIdsOrdered;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back();
        deg0vertices.pop_back();
        nodeIdsOrdered.push_back(v);
        for (int w : outgoingEdges[v]) {
            --indegree[w];
            if (indegree[w] == 0) {
                deg0vertices.push_back(w);
            }
        }
    }
    return nodeIdsOrdered;
}


int Instance::getMaxTensorParallelDegree() const {
    // returns max t such that some node has some TMPC for tensor-parallel degree t
    int res = 1;
    for (const Node &n : nodes) {
        for (const auto &it : n.TMPCs) {
            res = max(res, it.first);
        }
    }
    return res;
}


struct ResultStage {
    vector<int> nodes; // node ids (this is actually superfluous because of TMPCids)
    int dataParallelDegree; // d_i
    int tensorParallelDegree; // t_i
    unordered_map<int,string> TMPCids; // TMPCids[v] = identifier of the TMPC used for node v
};


struct Result {
    vector<ResultStage> stages;
    string debugInfo;
};


void to_json (json &j, const ResultStage &rs) {
    j = json{{"nodes", rs.nodes},
             {"dataParallelDegree", rs.dataParallelDegree},
             {"tensorParallelDegree", rs.tensorParallelDegree},
             {"TMPCids", rs.TMPCids}
    };
}


void to_json (json &j, const Result &r) {
    j = json{{"stages", r.stages}};
}


struct Graph {
    const Instance &ins; // already renumbered (nodes 0,1,2,...)
    const int boundOnS; // set as min(mbsInBatch, maxDevices)

    vector<vector<pair<int,double>>> incomingEdges; // v -> vector of {u,c(u,v)}
    vector<vector<pair<int,double>>> outgoingEdges; // v -> vector of {w,c(v,w)}
    vector<const Node*> node; // node[v] = pointer to node with (new) id v

    // ideals, represented as indicator vectors
    unordered_map<vector<bool>,int> idealToId; // maps ideal to its ID
    vector<vector<bool>> ideals; // maps ID to ideal
    vector<int> idealsSortedBySize; // IDs of ideals, sorted by size

    // NOTE: throughout the code, we are actually dealing with DOWNSETS, not ideals. should Ctrl+Replace

    // pairs of ideals (that induce contiguous sets)
    vector<vector<int>> immediateSubIdeals;
    // immediateSubIdeals[id] = IDs of ideals that are immediate subsets of the ideal with ID id
    vector<vector<int>> subIdeals;
    // subideals[id] = IDs of ideals that are subsets of the ideal with ID id
    // (this takes O(numberOfIdealPairs) space; could be done on the fly in the DP,
    //  but if one can't afford this memory then probably one can't afford the DP alg timewise)
    long long numberOfIdealPairs;

    Graph (const Instance &_ins);
    void generateIdeals ();
    void growIdeal (const vector<bool> &ideal, int myId);
    void prepareSubIdeals ();

    vector<bool> getContiguousSet (int id, int subId) const;
    vector<vector<vector<double>>> getAllTPSsForIdealPair (int id, int subId);

    double getDataParallelResyncCost (double parameterSize, int d) const;

    // reconstruction stuff:
    ResultStage getTPSWitnessFor (int id, int subId, int t, int d, int s, double targetTPS);
    // some "global" variables used in the reconstruction phase
    bool reconstructionModeForGetAllTPSsForIdealPair;
    bool reconstructionModeForSolveKnapsack;
    int reconstructionT, reconstructionD, reconstructionS, reconstructionY;
    double reconstructionTPS;
    vector<int> reconstructionTMPCindices; // output of solveKnapsack in reconstruction phase
    ResultStage reconstructionRS; // output of getTPSWitnessFor in reconstruction phase

    vector<double> solveKnapsack (const vector<vector<double>>& TMPCTPS,
                                  const vector<vector<double>>& TMPCMemoryUsageA,
                                  const vector<vector<double>>& TMPCMemoryUsageB,
                                  int maxY);
    
    vector<vector<vector<double>>> dp; // dp table
    int idOfFullSet;
    Result runDP();

    void renumberResultBack (Result &r) const;

    double getTPSForResult (const Result &r) const;
};


Graph::Graph (const Instance &_ins) :
    ins(_ins),
    boundOnS(min(_ins.mbsInBatch,_ins.maxDevices)),
    numberOfIdealPairs(0),
    reconstructionModeForGetAllTPSsForIdealPair(false),
    reconstructionModeForSolveKnapsack(false) {

    // precompute incomingEdges and outgoingEdges
    incomingEdges.resize(ins.nodes.size());
    outgoingEdges.resize(ins.nodes.size());
    for (const Edge &e : ins.edges) {
        incomingEdges[e.destId].emplace_back(e.sourceId, e.communicationCost);
        outgoingEdges[e.sourceId].emplace_back(e.destId, e.communicationCost);
    }
    // precompute node
    node.resize(ins.nodes.size());
    for (const Node &n : ins.nodes) {
        node[n.id] = &n;
    }

    generateIdeals();
    // immediateSubIdeals is prepared. now prepare subIdeals
    prepareSubIdeals();
}


void Graph::generateIdeals () {
    if (!ideals.empty()) {
        fail("generating ideals twice?");
    }

    // start with empty set
    const vector<bool> emptySet(ins.nodes.size(), false);
    idealToId[emptySet] = 0;
    ideals.push_back(emptySet);
    immediateSubIdeals.emplace_back();
    growIdeal(emptySet, 0);

    dbg << ideals.size() << " ideals" << endl;
    if (ideals.size() > IDEALS_LIMIT) {
        fail("too many ideals (current limit set at " + to_string(IDEALS_LIMIT) + "); this isn't going to work...");
    }

    // prepare idealsSortedBySize
    vector<pair<int,int>> sorter; // {<size,ideal id>}
    for (int i = 0; i < ideals.size(); ++i) {
        sorter.emplace_back(count(ideals[i].begin(), ideals[i].end(), true), i);
    }
    sort(sorter.begin(), sorter.end());
    for (const auto &it : sorter) {
        idealsSortedBySize.push_back(it.second);
    }
    assert(idealsSortedBySize[0] == 0);
}


void Graph::growIdeal (const vector<bool> &ideal, int myId) {
    // myId == idealToId[ideal]
    // try to add every vertex
    for (int v = 0; v < ins.nodes.size(); ++v) {
        if (!ideal[v]) {
            // try ideal+v as a new ideal
            // check if valid: do all v's successors belong to ideal?
            bool valid = true;
            for (const pair<int,double>& p : outgoingEdges[v]) {
                // v -> p.first
                if (!ideal[p.first]) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                vector<bool> newIdeal = ideal;
                newIdeal[v] = true;
                // check if newIdeal had already been generated
                if (!idealToId.count(newIdeal)) {
                    int newId = ideals.size();
                    idealToId[newIdeal] = newId;
                    ideals.push_back(newIdeal);
                    if (ideals.size() >= IDEALS_EXPLORATION_LIMIT) {
                        fail("already over " + to_string(IDEALS_EXPLORATION_LIMIT) + " ideals. this isn't going to work...");
                    }
                    immediateSubIdeals.emplace_back();
                    growIdeal(newIdeal, newId);
                }
                immediateSubIdeals[idealToId[newIdeal]].push_back(myId);
            }
        }
    }
}


void Graph::prepareSubIdeals () {
    // subideals = transitive closure of immediateSubIdeals

    numberOfIdealPairs = 0;
    subIdeals.resize(ideals.size());

    for (int id = 0; id < ideals.size(); ++id) {
        // we will generate subIdeals[id] using some BFS/DFS
        vector<int> queue = {id};
        unordered_set<int> enqueuedIdeals = {id};
        while (!queue.empty()) {
            int subId = queue.back();
            queue.pop_back();

            // now visiting subId
            if (subId != id) {
                subIdeals[id].push_back(subId);
                ++numberOfIdealPairs;
            }

            // expand further from subId
            for (int subSubId : immediateSubIdeals[subId]) {
                if (enqueuedIdeals.insert(subSubId).second == true) {
                    // subSubId was not in enqueuedIdeals before
                    queue.push_back(subSubId);
                }
            }
        }
    }

    dbg << numberOfIdealPairs << " ideal pairs" << endl;
}


// returns the difference ideals[id] \ ideals[subId] as vector<bool>
vector<bool> Graph::getContiguousSet (int id, int subId) const {
    vector<bool> ideal = ideals[id], subIdeal = ideals[subId];
    for (int v = 0; v < ins.nodes.size(); ++v) {
        if (subIdeal[v]) {
            ideal[v] = false;
        }
    }
    return ideal;
}


// returns the cost per sample (microbatch), in bytes
// (so this should be still divided by the bandwidth)
double Graph::getDataParallelResyncCost (double parameterSize, int d) const {
    return 4 * (d-1) * parameterSize / d / ins.mbsInBatch;
    // the factor 4 is following the implementation of the PipeDream planner
    // (modeling a distributed parameter server implementation of AllReduce)
}


Result Graph::runDP () {
    // initialize DP table: dp[ideal][k][s]
    // (partition ideal over <= k machines, sum-of-dp-degrees <= s)
    // note: both are AT MOST rather than EQUAL as in the paper
    dp.assign(ideals.size(), vector<vector<double>>(
              ins.maxDevices+1, vector<double>(
              boundOnS+1, INFTY)));

    // case of the empty set (ideal with ID 0)
    // we initialize all dp[0][*][*] = 0 so that it is monotone wrt k and s
    for (int k = 0; k <= ins.maxDevices; ++k) {
        for (int s = 0; s <= boundOnS; ++s) {
            dp[0][k][s] = 0;
        }
    }

    // dpDrops[id][k] will be the list of s such that dp[id][k][s] < dp[id][k][s-1]
    // (used for FASTER_DP_IMPLEMENTATION)
    vector<vector<vector<int>>> dpDrops(ideals.size(), vector<vector<int>>(
                                        ins.maxDevices+1, vector<int>(1, 0))); // dpDrops[id][k] = {0}

    // profiling stuff
    double timeSpentInGetAllTPSsForIdealPair = 0.0, timeSpentInDPLoop = 0.0;

    // here we go!
    dbg << "running DP..." << endl;
    for (int id : idealsSortedBySize) {
        if (id == 0) continue;
        // we want to fill dp[id][*][*] (already initialized to INFTY).
        // we will loop over every subideal subId (their list is already
        // precomputed in subIdeals[id] for convenience) and account for its
        // contributions to dp[id][*][*]
        for (int subId : subIdeals[id]) {

            clock_t startTime = clock();
            const vector<vector<vector<double>>> TPS = getAllTPSsForIdealPair(id, subId);
            timeSpentInGetAllTPSsForIdealPair += (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;
            // TPS[t][d][s] = min TPS when t-tensor and d-data-parallel partitioning
            //                id\subId (across t*d devices), with sum-of-dp-degrees <= s

            startTime = clock();
            for (int t = 1; t <= ins.maxDevices; ++t) if (!TPS[t].empty()) {
                for (int d = 1; d*t <= ins.maxDevices; ++d) if (ins.mbsInBatch % d == 0) {
                    for (int k = d*t; k <= ins.maxDevices; ++k) {
                        // id/subId on t*d devices, subId on k-t*d devices

                        // now we need to perform the updates of dp[id][k][s] for all s

                        if (!FASTER_DP_IMPLEMENTATION) {

                            // straightforward implementation

                            #pragma GCC unroll 16
                            for (int s = d; s <= boundOnS; ++s) {
                                minify(dp[id][k][s], max(dp[subId][k-t*d][s-d], TPS[t][d][s]));
                            }

                        } else { // FASTER_DP_IMPLEMENTATION

                            // we do not update dp[id][k][s] for all s here, but only for
                            // a selected subset that is sufficient
                            // (namely, those s where dp[subId][k-t*d][s-d] has decreased (from [s-1-d]))
                            // then, these updates will be propagated right after the loop over subId
                            // in order to ensure monotonicity

                            // dp[subId][k-t*d][s-d] is non-increasing wrt s
                            // TPS[t][d][s] is non-decreasing wrt s
                            // 1. find (using binary search) max index u >= 0 such that:
                            //    - f = dpDrops[subId][k-t*d][u] exists
                            //    - f+d <= boundOnS
                            //    - dp[subId][k-t*d][f] > TPS[t][d][f+d]
                            // 2. do minify from 0 to u incl. (use just dp, not TPS)
                            // 3. also do minify for u+1 if it is valid (two first points above) (use both dp and TPS here)
                            int u = -1, uf = 0, ut = int(dpDrops[subId][k-t*d].size()) - 1;
                            while (uf <= ut) {
                                int mid = (uf + ut) / 2;
                                int f = dpDrops[subId][k-t*d][mid];
                                if (f+d <= boundOnS && dp[subId][k-t*d][f] > TPS[t][d][f+d]) {
                                    u = mid;
                                    uf = mid+1;
                                } else {
                                    ut = mid-1;
                                }
                            }
                            #pragma GCC unroll 16
                            for (int v = 0; v <= u; ++v) {
                                const int f = dpDrops[subId][k-t*d][v];
                                minify(dp[id][k][f+d], dp[subId][k-t*d][f]);
                            }
                            if (u+1 < dpDrops[subId][k-t*d].size()) {
                                int f = dpDrops[subId][k-t*d][u+1];
                                if (f+d <= boundOnS) {
                                    assert(dp[subId][k-t*d][f] <= TPS[t][d][f+d]);
                                    minify(dp[id][k][f+d], TPS[t][d][f+d]);
                                }
                            }

                        }
                    }
                }
            }
            timeSpentInDPLoop += (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;
        }

        // ensure monotonicity
        // (only really needed if FASTER_DP_IMPLEMENTATION)
        for (int k = 0; k <= ins.maxDevices; ++k) {
            //assert(dpDrops[id][k] == vector<int>(1,0));
            for (int s = 0; s <= boundOnS; ++s) {
                if (k > 0) {
                    minify(dp[id][k][s], dp[id][k-1][s]);
                }
                if (s > 0) {
                    minify(dp[id][k][s], dp[id][k][s-1]);
                    if (dp[id][k][s] + 1e-9 < dp[id][k][s-1]) {
                        // significant drop
                        dpDrops[id][k].push_back(s);
                    }
                }
            }
        }
    }

    DBG(timeSpentInGetAllTPSsForIdealPair)
    DBG(timeSpentInDPLoop)

    // the solution (final TPS) is now known: it's max_k max_s dp[all nodes][k][s]

    // ID of ideal that contains all nodes
    idOfFullSet = idealToId.at(vector<bool>(ins.nodes.size(), true));

    double finalTPS = INFTY;
    int devicesUsed = -1, sUsed = -1;
    for (int k = 0; k <= ins.maxDevices; ++k) {
        for (int s = 1; s <= boundOnS; ++s) {
            if (dp[idOfFullSet][k][s] + 1e-9 < finalTPS) {
                finalTPS = dp[idOfFullSet][k][s];
                devicesUsed = k;
                sUsed = s;
            }
        }
    }
    if (finalTPS > INFTY/2) {
        // cannot partition the graph feasibly (in terms of memory usage)
        return Result(); // empty result
    }
    dbg << "max load = " << finalTPS << " using " << devicesUsed
        << " out of " << ins.maxDevices << " devices, and using sum-of-dp-degrees "
        << sUsed << " (batch size = " 
        << ins.mbsInBatch << ")" << endl;
    // note: the reported number of devices and batch size might possibly be overshot
    // (since we initialized all dp[0][*][*] = 0 at the beginning,
    //  and also since, if FASTER_DP_IMPLEMENTATION,
    //  we do minify(dp[id][k][s], dp[id][k-1][s]) and minify(dp[id][k][s], dp[id][k][s-1]))
    // however, the strict inequality in the comparison above should prevent this,
    // as this way we take the minimal (k,s) that attains this TPS

    // for debug/experiments only
    vector<int> transformerIds = ins.getTransformerIds();

    // now we reconstruct the solution
    Result result;
    int curId = idOfFullSet, curK = devicesUsed, curS = sUsed;
    while (curId != 0) { // curId is not empty set
        assert(curK > 0);
        assert(curS > 0);
        // how does dp[curId][curK][curS] arise?
        bool found = false;
        for (int subId : subIdeals[curId]) {
            const vector<vector<vector<double>>> TPS = getAllTPSsForIdealPair(curId, subId);
            // possible optimization: could only ask for s = curS

            for (int t = 1; t <= curK; ++t) if (!TPS[t].empty()) {
                for (int d = 1; t*d <= curK && d <= curS; ++d) {
                    if (ins.mbsInBatch % d != 0) continue; // not really necessary
                    // curId\subId on t*d devices, subId on curK-t*d devices
                    if (1e-9 > abs(dp[curId][curK][curS] - max(dp[subId][curK-t*d][curS-d], TPS[t][d][curS]))) {
                        // found the next stage
                        found = true;

                        ResultStage rs = getTPSWitnessFor(curId, subId, t, d, curS, TPS[t][d][curS]);
                        result.stages.push_back(rs);

                        assert(rs.dataParallelDegree == d);
                        assert(rs.tensorParallelDegree == t);

                        dbg << "formed a stage with nodes [" << rs.nodes << "] using d=" 
                            << rs.dataParallelDegree << " and t=" << rs.tensorParallelDegree
                            << " yielding TPS = " << TPS[t][d][curS] << endl;

                        int countTransformers = 0, countTransformersWithAR = 0;
                        for (const pair<int,string> &p : rs.TMPCids) {
                            if (count(transformerIds.begin(), transformerIds.end(), p.first)) {
                                ++countTransformers;
                                if (p.second == "activation recomp") {
                                    ++countTransformersWithAR;
                                }
                            }
                        }
                        dbg << "transformer layers (with act. recomp. / total) = "
                            << countTransformersWithAR << "/"
                            << countTransformers << endl << endl;

                        curS = curS - d;
                        curK = curK - t*d;
                        curId = subId;

                        break;
                    }
                }
                if (found) break;
            }
            if (found) break;
        }
        if (!found) {
            fail("didn't find any reconstruction step to make?");
        }
    }
    if (curK > 0 || curS > 0) {
        fail("k or s didn't fall to 0 by the end of reconstruction?");
    }

    const double verificationTPS = getTPSForResult(result);
    if (abs(finalTPS - verificationTPS) > 1e-5) {
        DBG(finalTPS)
        DBG(verificationTPS)
        fail("verification TPS is different");
    }

    // note: the result is in terms of the new numbers; if wanting to print it etc.,
    // then at the end, translate result from new numbers to old by calling:
    // renumberResultBack(result);

    return result;
}


// TPS[t][d][s] = min TPS when t-tensor and d-data-parallel partitioning
//                id\subId (across t*d devices), with sum-dp-degrees <= s
// TPS[t] can be empty for some t!
// (note: this should be deterministic as it will be rerun during reconstruction)
vector<vector<vector<double>>> Graph::getAllTPSsForIdealPair (int id, int subId) {
    const vector<bool> subgraphVB = getContiguousSet(id, subId);
    const vector<int> subgraph = vectorOfSetBits(subgraphVB);

    vector<vector<vector<double>>> result(ins.maxDevices+1);

    // t = degree of tensor parallelism
    for (int t = 1; t <= ins.maxDevices; ++t) {
        if (t > 1 && !TENSOR_PARALLELISM_ALLOWED) {
            break;
        }

        bool someNodeHasNoTMPCsForT = false;
        for (int v : subgraph) {
            if (!node[v]->TMPCs.count(t) || node[v]->TMPCs.at(t).empty()) {
                someNodeHasNoTMPCsForT = true;
                break;
            }
        }
        if (someNodeHasNoTMPCsForT) {
            // tensor parallelism of degree t is not supported
            // for some node of this subgraph
            continue;
        }

        // initialize the result vector
        result[t].assign(ins.maxDevices/t + 1, vector<double>(boundOnS+1, INFTY));

        vector<const vector<TMPC>*> TMPCs; // TMPCs[i] = TMPCs for node subgraph[i] on t devices
        vector<vector<double>> TMPCEdgeCommCosts;
        // TMPCEdgeCommCosts[i][l] = sum of all edge-associated communication costs
        //                           for the l-th TMPC of node subgraph[i] (on t devices)

        // prepare TMPCs and TMPCEdgeCommCosts
        for (int v : subgraph) {
            TMPCs.push_back(&(node[v]->TMPCs.at(t)));
            TMPCEdgeCommCosts.emplace_back();
            for (const TMPC &tmpc : *(TMPCs.back())) {
                double tmpcEdgeCommCost = 0.0;
                // need to take into account c(u,v) and sfw(u,v)
                // (first is a property of the edge, second is a property of the TMPC)
                // for all edges (u,v) incoming into v from outside S
                for (const pair<int,double> &p : incomingEdges[v]) {
                    // edge p.first -> v, of cost p.second
                    if (!subgraphVB[p.first]) {
                        tmpcEdgeCommCost += 2 * (p.second + tmpc.syncTimeFw.at(p.first));
                    }
                }

                // instead of "tmpc.syncTimeFw.at(p.first)" we could do this:
                // for (const pair<int,double> &p : tmpc.syncTimeFw) {
                //     if (!subgraphVB[p.first]) {
                //         tmpcEdgeCommCost += p.second;
                //     }
                // }
                // and then we could even turn syncTimeFw/Bw into vectors

                // and similarly for outgoing edges (c and sbw)
                for (const pair<int,double> &p : outgoingEdges[v]) {
                    // edge v -> p.first, of cost p.second
                    if (!subgraphVB[p.first]) {
                        tmpcEdgeCommCost += 2 * (p.second + tmpc.syncTimeBw.at(p.first));
                    }
                }
                TMPCEdgeCommCosts.back().push_back(tmpcEdgeCommCost);
            }
        }
        // TMPCs and TMPCEdgeCommCosts prepared

        // d = degree of data parallelism
        for (int d = 1; d*t <= ins.maxDevices; ++d) if (ins.mbsInBatch % d == 0) {

            if (d > 1 && !DATA_PARALLELISM_ALLOWED) {
                break;
            }

            // at this point, we have multiple TMPCs for each node
            // and we have to select one TMPC per node
            // (for each s; can be different TMPC-sets for different s;
            //  but only memory usage depends on s)

            vector<vector<double>> TMPCTotalTPSContribution, TMPCMemoryUsageA, TMPCMemoryUsageB;
            // TMPCTotalTPSContribution[i][l] = sum of all contributions to TPS (compute, comm)
            //    for the l-th TMPC of node subgraph[i] (on t-tensor and d-data parallelism)
            // similarly TMPCMemoryUsageA, B
            // prepare these:
            for (int i = 0; i < subgraph.size(); ++i) {
                const int v = subgraph[i];
                TMPCTotalTPSContribution.emplace_back();
                TMPCMemoryUsageA.emplace_back();
                TMPCMemoryUsageB.emplace_back();
                for (int l = 0; l < TMPCs[i]->size(); ++l) {
                    const TMPC &tmpc = (*TMPCs[i])[l];
                    TMPCMemoryUsageA.back().push_back(tmpc.memoryUsageA);
                    TMPCMemoryUsageB.back().push_back(tmpc.memoryUsageB);
                    // add up all the contributions of node v to the TPS:
                    const double communicationInBytes = 
                    // 1. edge-related communication
                        TMPCEdgeCommCosts[i][l] / d
                        +
                    // 2. data parallelism resync communication
                        getDataParallelResyncCost(tmpc.parameterSize, d);
                    // 3. compute
                    const double compute = tmpc.timePerSample / d;
                    const double totalTPSContribution = communicationInBytes / ins.bandwidth + compute;
                    TMPCTotalTPSContribution.back().push_back(totalTPSContribution);

                    if ((!ACTIVATION_RECOMPUTATION_ALLOWED && tmpc.id == "activation recomp")
                       || (ACTIVATION_RECOMPUTATION_FORCED && tmpc.id == "vanilla")) {
                        // only used for our specific experiments
                        // make this TMPC super unattractive
                        TMPCMemoryUsageB.back().back() = 1e60;
                        TMPCTotalTPSContribution.back().back() = 1e28;
                    }
                }
            }

            // we have now built a "knapsack" instance (well, one for each y = ceil(s/d))
            vector<double> knapsackResults = solveKnapsack(TMPCTotalTPSContribution,
                                                           TMPCMemoryUsageA,
                                                           TMPCMemoryUsageB,
                                                           boundOnS / d + 1);
            // knapsackResults[y] = best TPS (over all choices of TMPC-per-node)
            // when t-tensor- and d-data-parallel partitioning
            // id\subId (across t*d devices), with sum-of-dp-degrees <= d*y
            for (int s = 0; s <= boundOnS; ++s) {
                result[t][d][s] = knapsackResults[ceildiv(s, d)];
            }

            // stuff for the reconstruction phase
            if (reconstructionModeForGetAllTPSsForIdealPair) {
                reconstructionY = ceildiv(reconstructionS, reconstructionD);
                if (t == reconstructionT && d == reconstructionD) {
                    // if reconstructionTPS == knapsackResults[reconstructionY]:
                    if (1e-9 > abs(reconstructionTPS - knapsackResults[reconstructionY])) {
                        // got it
                        // now we need to build our "output", which is reconstructionRS
                        // (make sure to clear or replace each field, since reconstructionRS can be dirty)

                        reconstructionRS.nodes = subgraph;
                        reconstructionRS.dataParallelDegree = d;
                        reconstructionRS.tensorParallelDegree = t;
                        
                        // rerun knapsackResults in reconstruction mode
                        reconstructionModeForSolveKnapsack = true;
                        // this will fill reconstructionTMPCindices:
                        solveKnapsack(TMPCTotalTPSContribution,
                                      TMPCMemoryUsageA,
                                      TMPCMemoryUsageB,
                                      boundOnS / d + 1);
                        // now we need to translate this to reconstructionRS.TMPCids
                        reconstructionRS.TMPCids.clear();
                        for (int i = 0; i < subgraph.size(); ++i) {
                            const int v = subgraph[i];
                            reconstructionRS.TMPCids[v] = (*TMPCs[i])[reconstructionTMPCindices[i]].id;
                        }

                        reconstructionModeForSolveKnapsack = false;
                        // might as well return now
                        reconstructionModeForGetAllTPSsForIdealPair = false;
                    }
                }
            }
        }
    }

    return result;
}


// this function aims to solve the following knapsack problem variant for each y=1,...,maxY.
// we have n = TMPCTPS.size() nodes.
// for each node, we have some number of TMPCs; each has TMPCTPS, TMPCMemoryUsageA, TMPCMemoryUsageB.
// select one TMPC per node so that the sum of TMPCMemoryUsageA*y + TMPCMemoryUsageB <= ins.maxMemoryPerDevice
// and so that the sum of TMPCTPS is minimized.
// return that minimum.
// (note: this should be deterministic as it will be rerun during reconstruction)
vector<double> Graph::solveKnapsack (const vector<vector<double>>& TMPCTPS,
                                     const vector<vector<double>>& TMPCMemoryUsageA,
                                     const vector<vector<double>>& TMPCMemoryUsageB,
                                     int maxY) {
    const int n = TMPCTPS.size();
    vector<double> result(maxY+1, INFTY);

    // we will just do things independently for each y.
    // perhaps we can do something smarter in the future

    vector<int> fastestIndex(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int l = 1; l < TMPCTPS[i].size(); ++l) {
            if (TMPCTPS[i][fastestIndex[i]] > TMPCTPS[i][l]) {
                fastestIndex[i] = l;
            }
        }
    }

    vector<vector<double>> TMPCMemoryUsage = TMPCMemoryUsageB; // just needed to give it the right shape

    for (int y = 1; y <= maxY; ++y) {
        // update TMPCMemoryUsage
        for (int i = 0; i < n; ++i) {
            for (int l = 0; l < TMPCTPS[i].size(); ++l) {
                TMPCMemoryUsage[i][l] = TMPCMemoryUsageA[i][l] * y + TMPCMemoryUsageB[i][l];
            }
        }

        // check whether choosing the fastest TMPC everywhere might give a memory-feasible result
        // (then we'd just return it)
        double memoryOfFastestSolution = 0.0;
        for (int i = 0; i < n; ++i) {
            memoryOfFastestSolution += TMPCMemoryUsage[i][fastestIndex[i]];
        }
        if (memoryOfFastestSolution <= ins.maxMemoryPerDevice) {
            // cool! just use that
            result[y] = 0.0;
            for (int i = 0; i < n; ++i) {
                result[y] += TMPCTPS[i][fastestIndex[i]];
            }

            // reconstruction stuff (unfortunately the code repeats)
            if (reconstructionModeForSolveKnapsack) {
                if (reconstructionY == y) {
                    // fill reconstructionTMPCindices
                    reconstructionTMPCindices = fastestIndex;
                    dbg << "we picked the all-fastest knapsack solution, mem usage = " 
                        << memoryOfFastestSolution << endl;
                }
            }

            continue;
        }

        // or if choosing the lowest-memory TMPC everywhere is not even memory-feasible
        // (then there is no feasible solution)
        vector<int> lowestMemoryIndex(n, 0);
        for (int i = 0; i < n; ++i) {
            for (int l = 1; l < TMPCTPS[i].size(); ++l) {
                if (TMPCMemoryUsage[i][lowestMemoryIndex[i]] > TMPCMemoryUsage[i][l]) {
                    lowestMemoryIndex[i] = l;
                }
            }
        }
        double memoryOfLowestMemorySolution = 0.0;
        for (int i = 0; i < n; ++i) {
            memoryOfLowestMemorySolution += TMPCMemoryUsage[i][lowestMemoryIndex[i]];
        }
        if (memoryOfLowestMemorySolution > ins.maxMemoryPerDevice) {
            // no feasible solution. and there won't be one for larger y, either. so break
            break;
        }

        // the instance is non-trivial. now we run a heuristic
        vector<int> H = fastestIndex; // H[i] = index of the currently chosen TMPC for node i
        // we start from the all-fastest solution
        double memoryToShaveOff = memoryOfFastestSolution - ins.maxMemoryPerDevice;
        // we need to reduce the memory usage by this much

        if (ACTIVATION_RECOMPUTATION_ALL_LAYERS_OR_NONE) {
            // special thing for our experiments

            // at this point we know that the all-vanilla solution is not feasible
            // but the all-AC one is. so just return the latter
            H = lowestMemoryIndex;
            memoryToShaveOff = memoryOfLowestMemorySolution - ins.maxMemoryPerDevice;
        } else {
            // normal execution

            if (!KNAPSACK_FAST_HEURISTIC) {

                while (memoryToShaveOff > 1e-9) { // # iterations <= total number of TMPCs
                    // we do so by repeatedly picking the best-bang-for-buck swap (change of some H[i])
                    double leastBuckPerBang = INFTY;
                    int bestI = -1, bestL = -1;
                    // check all possible swaps
                    for (int i = 0; i < n; ++i) {
                        for (int l = 0; l < TMPCTPS[i].size(); ++l) {
                            // considering the change "H[i] := l"
                            const double memorySavings = TMPCMemoryUsage[i][H[i]] - TMPCMemoryUsage[i][l];
                            if (memorySavings > 1e-9) {
                                if (TMPCTPS[i][l] < TMPCTPS[i][H[i]]) {
                                    fail("we are gaining both on memory and TPS?!");
                                }
                                // important (somewhat): here we truncate large gains to what we actually need
                                const double bang = min(memorySavings, memoryToShaveOff);
                                const double buckPerBang = (TMPCTPS[i][l] - TMPCTPS[i][H[i]]) / bang;
                                if (leastBuckPerBang > buckPerBang) {
                                    leastBuckPerBang = buckPerBang;
                                    bestI = i;
                                    bestL = l;
                                }
                            }
                        }
                    }
                    if (bestI == -1) {
                        fail("no good change was found?!");
                    }
                    // now apply the best found change
                    const double memorySavings = TMPCMemoryUsage[bestI][H[bestI]] - TMPCMemoryUsage[bestI][bestL];
                    memoryToShaveOff -= memorySavings;
                    H[bestI] = bestL;
                }

            } else { // KNAPSACK_FAST_HEURISTIC

                // so far, for simplicity and speed, we have here
                // a version that only works for <= 2 TMPCs per (v,t)
                // (this is verified in checkInputCorrectness())

                vector<tuple<double,int,int>> sorter;
                for (int i = 0; i < n; ++i) {
                    for (int l = 0; l < TMPCTPS[i].size(); ++l) {
                        // considering the change "H[i] := l"
                        const double memorySavings = TMPCMemoryUsage[i][H[i]] - TMPCMemoryUsage[i][l];
                        if (memorySavings > 1e-9) {
                            if (TMPCTPS[i][l] < TMPCTPS[i][H[i]]) {
                                fail("we are gaining both on memory and TPS?!");
                            }
                            // here we don't truncate and might overshoot
                            // (see the similar point in the slower heuristic)
                            const double buckPerBang = (TMPCTPS[i][l] - TMPCTPS[i][H[i]]) / memorySavings;
                            sorter.emplace_back(buckPerBang, i, l);
                        }
                    }
                }
                sort(sorter.begin(), sorter.end());
                for (const auto &it : sorter) {
                    const int bestI = get<1>(it), bestL = get<2>(it);
                    // apply this change
                    const double memorySavings = TMPCMemoryUsage[bestI][H[bestI]] - TMPCMemoryUsage[bestI][bestL];
                    memoryToShaveOff -= memorySavings;
                    H[bestI] = bestL;
                    if (memoryToShaveOff < 1e-9) {
                        break; // done
                    }
                }

            }
        }
        // done - we have a feasible solution H
        result[y] = 0.0;
        for (int i = 0; i < n; ++i) {
            result[y] += TMPCTPS[i][H[i]];
        }

        // reconstruction stuff
        if (reconstructionModeForSolveKnapsack) {
            if (reconstructionY == y) {
                // fill reconstructionTMPCindices
                reconstructionTMPCindices = H;
                dbg << "nontrivial knapsack solution, mem usage = " << ins.maxMemoryPerDevice + memoryToShaveOff << endl;
            }
        }
    }

    // debug stuff for verification that our knapsack heuristic
    // almost always finds the optimal solution (see Appendix)
    if (OUTPUT_KNAPSACK_INSTANCES_FOR_INSPECTION) {
        static long long executionCount = 0;
        static ofstream knapsackFile("knapsacks.txt");
        ++executionCount;
        constexpr int y = 2;
        assert(y <= maxY);
        if (rand() % 3 == 0) {
            dbg << "executionCount = " << executionCount << endl;
            knapsackFile << setprecision(15) << fixed << ins.maxMemoryPerDevice << endl;
            knapsackFile << TMPCTPS.size() << endl;
            for (int i = 0; i < TMPCTPS.size(); ++i) {
                knapsackFile << TMPCTPS[i].size() << endl;
                for (int l = 0; l < TMPCTPS[i].size(); ++l) {
                    const double memUsageFor5 = y * TMPCMemoryUsageA[i][l] + TMPCMemoryUsageB[i][l];
                    knapsackFile << setprecision(15) << fixed << TMPCTPS[i][l] << " " << memUsageFor5 << endl;
                }
            }
            knapsackFile << setprecision(15) << fixed << result[y] << endl << endl;
        }
    }

    return result;
}


ResultStage Graph::getTPSWitnessFor (int id, int subId, int t, int d, int s, double targetTPS) {
    reconstructionModeForGetAllTPSsForIdealPair = true;
    reconstructionT = t;
    reconstructionD = d;
    reconstructionS = s;
    reconstructionTPS = targetTPS;
    getAllTPSsForIdealPair(id, subId); // this will fill reconstructionRS
    reconstructionModeForGetAllTPSsForIdealPair = false;
    return reconstructionRS;
}


void Graph::renumberResultBack (Result &r) const {
    for (ResultStage &rs : r.stages) {
        for (int &nodeId : rs.nodes) {
            nodeId = ins.oldNumber[nodeId];
        }
        unordered_map<int,string> newTMPCids;
        for (const pair<int,string> &p : rs.TMPCids) {
            newTMPCids[ins.oldNumber[p.first]] = p.second;
        }
        rs.TMPCids = move(newTMPCids);
    }
}


double Graph::getTPSForResult (const Result &r) const {
    // for sanity checks of returned solutions,
    // and also to judge baselines

    if (r.stages.empty()) {
        // infeasible/OOM/empty result
        return INFTY;
    }

    // first step: check that the solution is contiguous
    // (and that there is some topological order in the contracted graph)
    // and that every node belongs to exactly one subgraph
    // and that we don't use too many devices
    vector<int> stageOfNode(ins.nodes.size(), -1);
    int devicesUsed = 0, sumOfDpDegrees = 0;
    for (int i = 0; i < r.stages.size(); ++i) {
        for (int v : r.stages[i].nodes) {
            if (stageOfNode[v] != -1) {
                fail("duplicate node");
            }
            stageOfNode[v] = i;
        }
        if (r.stages[i].dataParallelDegree < 1 || r.stages[i].dataParallelDegree > ins.maxDevices) {
            fail("wrong data-parallel degree");
        }
        if (ins.mbsInBatch % r.stages[i].dataParallelDegree != 0) {
            fail("data-parallel degree must divide the number of microbatches in a batch");
        }
        if (r.stages[i].tensorParallelDegree < 1 || r.stages[i].tensorParallelDegree > ins.maxDevices) {
            fail("wrong tensor-parallel degree");
        }
        devicesUsed += r.stages[i].dataParallelDegree * r.stages[i].tensorParallelDegree;
        sumOfDpDegrees += r.stages[i].dataParallelDegree;
    }
    for (const Edge &e : ins.edges) {
        if (stageOfNode[e.sourceId] > stageOfNode[e.destId]) {
            fail("problem with contiguity (or stages given in wrong order)");
        }
    }
    for (int v = 0; v < ins.nodes.size(); ++v) {
        if (-1 == stageOfNode[v]) {
            fail("node does not appear in any subgraph");
        }
    }
    if (sumOfDpDegrees > ins.mbsInBatch) {
        fail("sum of data-parallel degrees too large");
    }
    if (devicesUsed > ins.maxDevices) { 
        fail("too many devices used");
    }

    for (int v = 0; v < ins.nodes.size(); ++v) {
        if (v != ins.nodes[v].id) {
            fail("some issue with numbering");
        }
    }

    // now we want to compute the TPS,
    // and also verify it's not OOM
    double finalTPS = 0.0;
    int suffixSumOfDataParallelDegrees = sumOfDpDegrees;
    for (int i = 0; i < r.stages.size(); ++i) {
        const ResultStage &rs = r.stages[i];

        // we want to compute TPS for this stage
        
        const int d = rs.dataParallelDegree, t = rs.tensorParallelDegree;

        const int y = ceildiv(suffixSumOfDataParallelDegrees, d);
        suffixSumOfDataParallelDegrees -= d; // update for next iteration

        // get TMPCs (also check TMPCids)
        unordered_map<int, const TMPC*> TMPCs;
        for (const pair<int,string> &p : rs.TMPCids) {
            const int v = p.first;
            if (!count(rs.nodes.begin(), rs.nodes.end(), v)) {
                fail("node appears in TMPCids but not in nodes");
            }
            if (v < 0 || v >= ins.nodes.size()) {
                fail("wrong node id in TMPCids");
            }
            if (TMPCs.count(v)) {
                fail("duplicate node in TMPCids");
            }
            bool found = false;
            if (!ins.nodes[v].TMPCs.count(t)) {
                fail("no TMPCs for that node and that t");
            }
            for (const TMPC &tmpc : ins.nodes[v].TMPCs.at(t)) {
                if (tmpc.id == p.second) {
                    assert(!found);
                    found = true;
                    TMPCs[v] = &tmpc;
                }
            }
            if (!found) {
                fail("no TMPC with that ID was found");
            }
        }
        if (rs.nodes.size() > rs.TMPCids.size()) {
            fail("more nodes in nodes than in TMPCids");
        }
        // okay, TMPCs is populated. now compute TPS and memory usage

        double compute = 0.0, edgeCommunicationInBytes = 0.0, dataParallelCommunicationInBytes = 0.0, 
               memoryUsage = 0.0;

        for (int v : rs.nodes) {
            compute += TMPCs[v]->timePerSample;
            for (const pair<int,double> &p : incomingEdges[v]) {
                if (!count(rs.nodes.begin(), rs.nodes.end(), p.first)) {
                    edgeCommunicationInBytes += 2 * (p.second + TMPCs[v]->syncTimeFw.at(p.first));
                }
            }
            for (const pair<int,double> &p : outgoingEdges[v]) {
                if (!count(rs.nodes.begin(), rs.nodes.end(), p.first)) {
                    edgeCommunicationInBytes += 2 * (p.second + TMPCs[v]->syncTimeBw.at(p.first));
                }
            }
            dataParallelCommunicationInBytes += getDataParallelResyncCost(TMPCs[v]->parameterSize, d);
            memoryUsage += TMPCs[v]->memoryUsageA * y + TMPCs[v]->memoryUsageB;
        }

        if (memoryUsage > ins.maxMemoryPerDevice) {
            // given solution is OOM
            return INFTY;
        }

        const double communicationInBytes = edgeCommunicationInBytes / d + dataParallelCommunicationInBytes;

        if (DEBUG_DATA_PARALLEL_COSTS) {
            const double dataParallelCost = dataParallelCommunicationInBytes / ins.bandwidth;
            const double theRestxxxxxxxxx = (edgeCommunicationInBytes / ins.bandwidth + compute) / d;
            cerr << setprecision(10) << fixed;
            DBG(dataParallelCost);
            DBG(theRestxxxxxxxxx);
        }

        const double stageTPS = communicationInBytes / ins.bandwidth + compute / d;

        finalTPS = max(finalTPS, stageTPS);
    }
    return finalTPS;
}


Result runPipeDream2BWPlanner (const Instance &ins,
                               bool useTensorParallelism,
                               bool tryPuttingNonTransformerNodesSeparately) {
    vector<int> transformers = ins.getTransformerIds();
    vector<int> initialNodes, finalNodes; // before and after transformers, respectively

    while (transformers.size() + initialNodes.size() + finalNodes.size() < ins.nodes.size()) {
        for (const Edge &e : ins.edges) {
            const int u = e.sourceId, v = e.destId;
            const bool initialU = count(initialNodes.begin(), initialNodes.end(), u);
            const bool initialV = count(initialNodes.begin(), initialNodes.end(), v);
            const bool finalU = count(finalNodes.begin(), finalNodes.end(), u);
            const bool finalV = count(finalNodes.begin(), finalNodes.end(), v);
            const bool transformerU = count(transformers.begin(), transformers.end(), u);
            const bool transformerV = count(transformers.begin(), transformers.end(), v);

            if (!initialU && !transformerU && !finalU) {
                if (initialV || transformerV) {
                    initialNodes.push_back(u);
                }
            }
            if (!initialV && !transformerV && !finalV) {
                if (transformerU || finalU) {
                    finalNodes.push_back(v);
                }
            }
        }
    }

    Graph g(ins); // needed to run getTPSForResult()

    double bestTPS = INFTY;
    Result bestResult;

    for (int putNonTransformerNodesSeparately = 0;
         putNonTransformerNodesSeparately <= tryPuttingNonTransformerNodesSeparately;
         ++putNonTransformerNodesSeparately) {

        for (int stages = 1 + 2 * putNonTransformerNodesSeparately;
             stages <= ins.maxDevices && stages <= transformers.size() + 2 * putNonTransformerNodesSeparately;
             stages ++) {

            Result result;
            result.stages.resize(stages);

            const int transformerStages = stages - 2 * putNonTransformerNodesSeparately,
                      firstTransformerStage = putNonTransformerNodesSeparately,
                      lastTransformerStage = stages - 1 - putNonTransformerNodesSeparately;
            assert(lastTransformerStage - firstTransformerStage + 1 == transformerStages);

            // divide transformers equally-ish among stages
            const int transformersPerStage = transformers.size() / transformerStages,
                      largerStages = transformers.size() % transformerStages;
            int nextTransformerIndex = 0;
            for (int st = firstTransformerStage; st <= lastTransformerStage; ++st) {
                // last `largerStages` stages have one additional transformer each
                const bool thisIsALargerStage = (st >= lastTransformerStage + 1 - largerStages);
                for (int i = 0; i < transformersPerStage + thisIsALargerStage; ++i) {
                    result.stages[st].nodes.push_back(transformers[nextTransformerIndex]);
                    nextTransformerIndex++;
                }
            }
            assert(nextTransformerIndex == transformers.size());
            // the initial nodes go to first stage, the final nodes go to last stage
            append(result.stages[0].nodes, initialNodes);
            append(result.stages.back().nodes, finalNodes);
            result.debugInfo = (putNonTransformerNodesSeparately ? "1" : "0");

            for (int d = 1; stages*d <= min(ins.maxDevices, ins.mbsInBatch); d ++) {
                if (ins.mbsInBatch % d == 0) {
                    for (int t = 1; stages*d*t <= ins.maxDevices; t ++) {
                        if (t > 1) {
                            if (!useTensorParallelism) {
                                break;
                            }
                            // should check if all nodes support t-degree tensor parallelism
                            // we'll just check one node since in our inputs they all support some t or not
                            if (!ins.nodes[0].TMPCs.count(t)) {
                                continue;
                            }
                        }

                        for (ResultStage &rs : result.stages) {
                            rs.dataParallelDegree = d;
                            rs.tensorParallelDegree = t;
                        }

                        // PipeDream-2BW uses activation recomputation everywhere or nowhere
                        for (bool activationRecomputationEverywhere : {false, true}) {
                            for (ResultStage &rs : result.stages) {
                                for (int v : rs.nodes) {
                                    rs.TMPCids[v] = activationRecomputationEverywhere ? "activation recomp" : "vanilla";
                                }
                            }

                            // result is built. try it
                            const double TPS = g.getTPSForResult(result);
                            if (TPS < bestTPS) {
                                bestTPS = TPS;
                                bestResult = result;
                            }
                        }
                    }
                }
            }
        }
    }

    if (bestTPS > INFTY/2) {
        dbg << "2BW couldn't partition in any way (OOM)" << endl;
        return bestResult; // empty result
    }

    dbg << "best 2BW: TPS = " << bestTPS
        << ", stages = " << bestResult.stages.size()
        << ", d = " << bestResult.stages[0].dataParallelDegree
        << ", t = " << bestResult.stages[0].tensorParallelDegree
        << ", activation recomp = " << bestResult.stages[0].TMPCids.begin()->second
        << ", put non-transformers separately = " << bestResult.debugInfo
        << endl;

    //for (const ResultStage &rs : bestResult.stages) {
    //    dbg << rs.nodes << endl;
    //}

    return bestResult;
}



// run the PipeDream-2BW-like planner,
// but without treating transformer and non-transformer layers differently
Result runPipeDream2BWPlannerNonTransformer (const Instance &ins,
                                             bool useTensorParallelism) {
    Graph g(ins); // needed to run getTPSForResult()

    const vector<int> nodeIds = ins.getNodeIdsToposorted();

    double bestTPS = INFTY;
    Result bestResult;

    for (int stages = 1; stages <= ins.maxDevices; ++stages) {
        Result result;
        result.stages.resize(stages);

        // divide layers equally-ish among stages
        const int layersPerStage = nodeIds.size() / stages,
                  largerStages = nodeIds.size() % stages;
        int nextStageIndex = 0;
        for (int st = 0; st <= stages - 1; ++st) {
            // last `largerStages` have one additional layer each
            const bool thisIsALargerStage = (st >= stages - largerStages);
            for (int i = 0; i < layersPerStage + thisIsALargerStage; ++i) {
                result.stages[st].nodes.push_back(nodeIds[nextStageIndex]);
                nextStageIndex++;
            }
        }
        assert(nextStageIndex == nodeIds.size());

        for (int d = 1; stages*d <= min(ins.maxDevices, ins.mbsInBatch); d ++) {
            if (ins.mbsInBatch % d == 0) {
                for (int t = 1; stages*d*t <= ins.maxDevices; t ++) {
                    if (t > 1) {
                        if (!useTensorParallelism) {
                            break;
                        }
                        // should check if all nodes support t-degree tensor parallelism
                        // we'll just check one node since in our inputs they all support some t or not
                        if (!ins.nodes[0].TMPCs.count(t)) {
                            continue;
                        }
                    }

                    for (ResultStage &rs : result.stages) {
                        rs.dataParallelDegree = d;
                        rs.tensorParallelDegree = t;
                    }

                    // PipeDream-2BW uses activation recomputation everywhere or nowhere
                    for (bool activationRecomputationEverywhere : {false, true}) {
                        for (ResultStage &rs : result.stages) {
                            for (int v : rs.nodes) {
                                rs.TMPCids[v] = activationRecomputationEverywhere ? "activation recomp" : "vanilla";
                            }
                        }

                        // result is built. try it
                        const double TPS = g.getTPSForResult(result);
                        if (TPS < bestTPS) {
                            bestTPS = TPS;
                            bestResult = result;
                        }
                    }
                }
            }
        }
    }


    if (bestTPS > INFTY/2) {
        dbg << "2BW couldn't partition in any way (OOM)" << endl;
        return bestResult; // empty result
    }

    dbg << "best 2BW: TPS = " << bestTPS
        << ", stages = " << bestResult.stages.size()
        << ", d = " << bestResult.stages[0].dataParallelDegree
        << ", t = " << bestResult.stages[0].tensorParallelDegree
        << ", activation recomp = " << bestResult.stages[0].TMPCids.begin()->second
        << endl;

    //for (const ResultStage &rs : bestResult.stages) {
    //    dbg << rs.nodes << endl;
    //}

    return bestResult;
}



// build a single configuration, one of those that the PipeDream-2BW-like planner would consider
// (here code unfortunately repeats a lot from runPipeDream2BWPlanner)
double buildEquiPartitionResult (Instance &ins,
                                 int d,
                                 int t,
                                 int stages,
                                 bool putNonTransformerNodesSeparately,
                                 bool activationRecomputationEverywhere) {
    vector<int> transformers = ins.getTransformerIds();
    vector<int> initialNodes, finalNodes; // before and after transformers, respectively

    while (transformers.size() + initialNodes.size() + finalNodes.size() < ins.nodes.size()) {
        for (const Edge &e : ins.edges) {
            const int u = e.sourceId, v = e.destId;
            const bool initialU = count(initialNodes.begin(), initialNodes.end(), u);
            const bool initialV = count(initialNodes.begin(), initialNodes.end(), v);
            const bool finalU = count(finalNodes.begin(), finalNodes.end(), u);
            const bool finalV = count(finalNodes.begin(), finalNodes.end(), v);
            const bool transformerU = count(transformers.begin(), transformers.end(), u);
            const bool transformerV = count(transformers.begin(), transformers.end(), v);

            if (!initialU && !transformerU && !finalU) {
                if (initialV || transformerV) {
                    initialNodes.push_back(u);
                }
            }
            if (!initialV && !transformerV && !finalV) {
                if (transformerU || finalU) {
                    finalNodes.push_back(v);
                }
            }
        }
    }

    Graph g(ins); // needed to run getTPSForResult()
    assert(stages >= 1 + 2 * putNonTransformerNodesSeparately);
    assert(stages <= ins.maxDevices && stages <= transformers.size() + 2 * putNonTransformerNodesSeparately);

    Result result;
    result.stages.resize(stages);

    const int transformerStages = stages - 2 * putNonTransformerNodesSeparately,
                firstTransformerStage = putNonTransformerNodesSeparately,
                lastTransformerStage = stages - 1 - putNonTransformerNodesSeparately;
    assert(lastTransformerStage - firstTransformerStage + 1 == transformerStages);

    // divide transformers equally-ish among stages
    const int transformersPerStage = transformers.size() / transformerStages,
                largerStages = transformers.size() % transformerStages;
    int nextTransformerIndex = 0;
    for (int st = firstTransformerStage; st <= lastTransformerStage; ++st) {
        // last `largerStages` stages have one additional transformer each
        const bool thisIsALargerStage = (st >= lastTransformerStage + 1 - largerStages);
        for (int i = 0; i < transformersPerStage + thisIsALargerStage; ++i) {
            result.stages[st].nodes.push_back(transformers[nextTransformerIndex]);
            nextTransformerIndex++;
        }
    }
    assert(nextTransformerIndex == transformers.size());
    // the initial nodes go to first stage, the final nodes go to last stage
    append(result.stages[0].nodes, initialNodes);
    append(result.stages.back().nodes, finalNodes);
    result.debugInfo = (putNonTransformerNodesSeparately ? "1" : "0");

    assert(d >= 1);
    assert(stages*d <= min(ins.maxDevices, ins.mbsInBatch));
    assert(t >= 1);
    assert(stages*d*t <= ins.maxDevices);
    assert(ins.mbsInBatch % d == 0);

    // should check if all nodes support t-degree tensor parallelism
    // we'll just check one node since in our inputs they all support some t or not
    assert(ins.nodes[0].TMPCs.count(t));

    for (ResultStage &rs : result.stages) {
        rs.dataParallelDegree = d;
        rs.tensorParallelDegree = t;
    }

    for (ResultStage &rs : result.stages) {
        for (int v : rs.nodes) {
            rs.TMPCids[v] = activationRecomputationEverywhere ? "activation recomp" : "vanilla";
        }
    }

    // result is built. try it
    const double TPS = g.getTPSForResult(result);

    // dbg output
    dbg << "TPS = " << TPS << endl;

    return TPS;
}



void runOurAlgoOnInstances (const vector<Instance> &instances) {
    for (const Instance &ins : instances) {
        // our alg
        clock_t startTime = clock();
        Graph g(ins);
        Result our = g.runDP();
        double runtime = (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;
        DBG(runtime)
        if (our.stages.empty()) {
            continue; // infeasible, not interesting
        }
        double ourTPS = g.getTPSForResult(our);
    }
}


// read the BERT32 instance, then possibly add more transformer layers
Instance readBERTA100 (int noTransformers) {
    assert(noTransformers >= 32);
    json j;
    ifstream jsonfile("inputs/bert32a100.json");
    jsonfile >> j;
    Instance ins = j.get<Instance>();
    for (int t = 33; t <= noTransformers; ++t) {
        ins.insertTransformerLayer();
    }
    vector<int> transformers = ins.getTransformerIds();
    dbg << "transformer (renumbered) ids = [" << transformers << "]" << endl;
    return ins;
}


Instance readGNMT () {
    json j;
    ifstream jsonfile("inputs/gnmt.json");
    jsonfile >> j;
    Instance ins = j.get<Instance>();
    return ins;
}


Instance readResnet () {
    json j;
    ifstream jsonfile("inputs/resnet.json");
    jsonfile >> j;
    Instance ins = j.get<Instance>();
    return ins;
}


// run just a single instance
void single () {
    Instance ins = readBERTA100(32);
    ins.maxMemoryPerDevice = 8.0 * (1 << 30);
    ins.maxDevices = 512;
    ins.mbsInBatch = 1920;
    ins.bandwidth = 25.0 * (1LL << 30);
    
    runOurAlgoOnInstances({ins});
}


void plots () {
    for (double memGB : {1,2,8,80}) {
        for (int bertSize : {32}) {
            for (int batchSize : {1920}) {
                stringstream ss;
                ss << "bert" << bertSize << "-" << memGB << "GB-bs" << batchSize << ".csv";
                ofstream of(ss.str());
                of << ",Number of devices,color,\\sf{type},\\sf{throughput}\n";
                int cnt = 0;

                dbg << "=================================================" << endl;
                dbg << "bertSize = " << bertSize << ", batch size = " << batchSize << endl;
                dbg << "=================================================" << endl;

                for (int k : {8,32,64,128,512,1024,2048}) {

                    dbg << "=================================================" << endl;
                    dbg << "now running k = " << k << endl;
                    dbg << "=================================================" << endl;

                    Instance ins = readBERTA100(bertSize);
                    ins.maxMemoryPerDevice = (1 << 30) * 1.0 * memGB;
                    ins.maxDevices = k;
                    ins.bandwidth = 25.0 * (1 << 30);
                    ins.mbsInBatch = batchSize;

                    if (ins.getMaxTensorParallelDegree() * batchSize < k) {
                        dbg << "skipping device count " << k << " since they cannot all be used" << endl;
                        continue;
                    }

                    Graph g(ins);

                    const Result ourResult = g.runDP();
                    if (ourResult.stages.empty()) {
                        dbg << "skipping device count " << k << " since our solution is OOM" << endl;
                        continue;
                    }
                    const double ourTPS = g.getTPSForResult(ourResult);
                    assert(ourTPS < INFTY/2);
                    of << (cnt++) << ",$k=" << k << "$,";
                    of << "#2b7bba,Piper,1.000\n";

                    DATA_PARALLELISM_ALLOWED = false;
                    const double noDP = g.getTPSForResult(g.runDP());
                    DATA_PARALLELISM_ALLOWED = true;
                    of << (cnt++) << ",$k=" << k << "$,";
                    of << "#DB7093,no DP," << setprecision(3) << fixed << (ourTPS / noDP) << "\n";

                    TENSOR_PARALLELISM_ALLOWED = false;
                    const double noTP = g.getTPSForResult(g.runDP());
                    TENSOR_PARALLELISM_ALLOWED = true;
                    of << (cnt++) << ",$k=" << k << "$,";
                    of << "#FFD700,no TP," << setprecision(3) << fixed << (ourTPS / noTP) << "\n";

                    ACTIVATION_RECOMPUTATION_ALLOWED = false;
                    const double noAR = g.getTPSForResult(g.runDP());
                    ACTIVATION_RECOMPUTATION_ALLOWED = true;
                    of << (cnt++) << ",$k=" << k << "$,";
                    of << "#008000,no AR," << setprecision(3) << fixed << (ourTPS / noAR) << "\n";

                    //const double PD2BW = g.getTPSForResult(runPipeDream2BWPlanner(ins, false, false));
                    //of << (cnt++) << ",$k=" << k << "$,";
                    //of << "#191970,equi-no TP-no sep," << setprecision(3) << fixed << (ourTPS / PD2BW) << "\n";

                    //const double PD2BWHR = g.getTPSForResult(runPipeDream2BWPlanner(ins, true, false));
                    //of << (cnt++) << ",$k=" << k << "$,";
                    //of << "#FF4500,equi-no sep," << setprecision(3) << fixed << (ourTPS / PD2BWHR) << "\n";

                    const double PD2BWsep = g.getTPSForResult(runPipeDream2BWPlanner(ins, false, true));
                    of << (cnt++) << ",$k=" << k << "$,";
                    of << "#191970,equi-no TP," << setprecision(3) << fixed << (ourTPS / PD2BWsep) << "\n";

                    const double PD2BWsepHR = g.getTPSForResult(runPipeDream2BWPlanner(ins, true, true));
                    of << (cnt++) << ",$k=" << k << "$,";
                    of << "#FF4500,equi," << setprecision(3) << fixed << (ourTPS / PD2BWsepHR) << "\n";
                }
            }
        }
    }
}


void correlationExperiment () {
    Instance ins = readBERTA100(32);
    ins.maxDevices = 64;
    ins.bandwidth = (300.0 + 25.0)/2 * (1 << 30);
    vector<double> results;
    ins.mbsInBatch = 128;
    results.push_back(buildEquiPartitionResult(ins, 32, 1, 2, false, true));
    results.push_back(buildEquiPartitionResult(ins, 16, 1, 4, false, true));
    results.push_back(buildEquiPartitionResult(ins, 8, 1, 8, false, true));
    results.push_back(buildEquiPartitionResult(ins, 4, 1, 16, false, true));
    results.push_back(buildEquiPartitionResult(ins, 2, 1, 32, false, true));
    ins.mbsInBatch = 512;
    results.push_back(buildEquiPartitionResult(ins, 32, 1, 2, false, true));
    results.push_back(buildEquiPartitionResult(ins, 16, 1, 4, false, true));
    results.push_back(buildEquiPartitionResult(ins, 8, 1, 8, false, true));
    results.push_back(buildEquiPartitionResult(ins, 4, 1, 16, false, true));
    results.push_back(buildEquiPartitionResult(ins, 2, 1, 32, false, true));
    results.push_back(-1e30);
    results.push_back(-1e30);
    results.push_back(-1e30);
    ins.mbsInBatch = 128;
    results.push_back(buildEquiPartitionResult(ins, 32, 2, 1, false, true));
    results.push_back(buildEquiPartitionResult(ins, 16, 4, 1, false, true));
    results.push_back(buildEquiPartitionResult(ins, 8, 8, 1, false, true));
    ins.mbsInBatch = 512;
    results.push_back(buildEquiPartitionResult(ins, 32, 2, 1, false, true));
    results.push_back(buildEquiPartitionResult(ins, 16, 4, 1, false, true));
    results.push_back(buildEquiPartitionResult(ins, 8, 8, 1, false, true));
    for (double res : results) {
        cout << res << endl;
    }
}





void scalability () {
    for (int bertSize : {32,48,64,96}) {
        for (int batchSize : {480,512,1920,2048}) {
            //if (bertSize == 96 && batchSize == 2048) continue;
            Instance ins = readBERTA100(bertSize);
            stringstream ss;
            ss << "bert" << bertSize << "bs" << batchSize << "times.txt";
            ofstream of(ss.str());
            of << "k time stddev\n";
            for (int k : {8,16,32,64,128,256,512,1024,1536,2048}) {
                const int tries = (k > 600) ? 3 : 5;
                vector<double> times;
                for (int t = 0; t < tries; ++t) {
                    ins.maxDevices = k;
                    ins.mbsInBatch = batchSize;
                    clock_t startTime = clock();
                    Graph g(ins);
                    g.runDP();
                    const double runtime = (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;
                    times.push_back(runtime);
                }
                of << k << " " << setprecision(3) << fixed << average(times) << " " << sampleStddev(times) << endl;
            }
        }
    }
}



void runAndWrite (Graph &g, const string &filenameCore) {
    g.runDP();

    ofstream ofDevices(filenameCore + "-devices.txt");
    ofDevices << "k TPS\n";
    for (int k = 8; k <= g.ins.maxDevices; ++k) {
        const double TPS_k = g.dp[g.idOfFullSet][k][g.boundOnS];
        if (TPS_k > INFTY/2) {
            continue;
        }
        ofDevices << k << " " << setprecision(7) << fixed << TPS_k << "\n";
    }

    ofstream ofSBound(filenameCore + "-sbound.txt");
    ofSBound << "s TPS\n";
    for (int s = 8; s <= g.boundOnS; ++s) {
        const double TPS_s = g.dp[g.idOfFullSet][g.ins.maxDevices][s];
        if (TPS_s > INFTY/2) {
            continue;
        }
        ofSBound << s << " " << setprecision(7) << fixed << TPS_s << "\n";
    }
}


// run the PipeDream-2BW-like planner
void runPD2BWAndWrite (Instance &ins, const string &filenameCore, bool useTensorParallelism, bool tryPuttingNonTransformerNodesSeparately) {

    const int backupMaxDevices = ins.maxDevices;

    ofstream ofDevices(filenameCore + "-devices.txt");
    ofDevices << "k TPS\n";
    for (int k : {8,16,32,64,128,256,512,768,1024,1024+256,1024+512,1024+512+256,2048}) {
        ins.maxDevices = k;
        Graph g(ins);
        ofDevices << k << " " << setprecision(7) << fixed << g.getTPSForResult(runPipeDream2BWPlanner(ins, useTensorParallelism, tryPuttingNonTransformerNodesSeparately)) << endl;
    }

    ins.maxDevices = backupMaxDevices;


    ofstream ofBatch(filenameCore + "-batch.txt");
    ofBatch << "bs TPS\n";
    for (int bs : {8,16,32,64,128,256,256+128,512,512+128,512+256,512+256+128,1024}) {
        ins.mbsInBatch = bs;
        Graph g(ins);
        ofBatch << bs << " " << setprecision(7) << fixed << g.getTPSForResult(runPipeDream2BWPlanner(ins, useTensorParallelism, tryPuttingNonTransformerNodesSeparately)) << endl;
    }
}


// measuring runtime
void parallelizabilityExperiment (int bertSize, int memSize, int batchSize) {
    Instance ins = readBERTA100(bertSize);
    ins.maxDevices = 2048;
    ins.maxMemoryPerDevice = memSize * 1.0 * (1 << 30);
    ins.mbsInBatch = batchSize;
    Graph g(ins);

    // Piper
    runAndWrite(g, "parallel-piper-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize));

    DATA_PARALLELISM_ALLOWED = false;
    runAndWrite(g, "parallel-nodp-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize));
    DATA_PARALLELISM_ALLOWED = true;

    TENSOR_PARALLELISM_ALLOWED = false;
    runAndWrite(g, "parallel-notp-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize));
    TENSOR_PARALLELISM_ALLOWED = true;

    ACTIVATION_RECOMPUTATION_ALLOWED = false;
    runAndWrite(g, "parallel-noar-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize));
    ACTIVATION_RECOMPUTATION_ALLOWED = true;

    //runPD2BWAndWrite(ins, "parallel-equi-no sep-no-TP-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize), false, false);
    //runPD2BWAndWrite(ins, "parallel-equi-no sep-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize), true, false);
    runPD2BWAndWrite(ins, "parallel-equi-no-TP-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize), false, true);
    runPD2BWAndWrite(ins, "parallel-equi-" + to_string(bertSize) + "-" + to_string(memSize) + "GB-bs" + to_string(batchSize), true, true);
}


void runResnet () {
    vector<pair<int,double>> pairs = {
        {8, 4.0}, {8, 8.0},
        {16, 1.5}, {16, 2.0}, {16, 4.0}, {16, 8.0},
        {32, 1.5}, {32, 2.0}, {32, 4.0}, {32, 8.0},
        {64, 1.0}, {64, 1.5}, {64, 2.0}, {64, 4.0},
        {128, 1.0}, {128, 1.5}, {128, 2.0}, {128, 4.0},
        {8, 16.0}, {16, 16.0}, {32, 16.0}, {64, 8.0}, {64, 16.0},
        {128, 8.0}, {128, 16.0}
    };
    sort(pairs.begin(), pairs.end());
    for (pair<int,double> mm : pairs) {
        cerr << endl << endl << endl << endl;
        DBG(mm.first);
        DBG(mm.second);
        Instance ins = readResnet();
        ins.maxMemoryPerDevice = mm.second * (1 << 30);
        ins.maxDevices = mm.first;
        //ins.mbsInBatch = 1920*4;
        ins.bandwidth = 25.0 * (1LL << 30);
        
        // our alg
        clock_t startTime = clock();
        Graph g(ins);
        Result our = g.runDP();
        double runtime = (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;
        DBG(runtime)
        if (our.stages.empty()) {
            continue; // infeasible, not interesting
        }
        double ourTPS = g.getTPSForResult(our);

        Result equi = runPipeDream2BWPlannerNonTransformer(ins, false);
        double equiTPS = equi.stages.empty() ? 1e30 : g.getTPSForResult(equi);
        double ratio = equi.stages.empty() ? 0 : ourTPS / equiTPS;

        cout << mm.first << " & "
             << mm.second << " & "
             << setprecision(3) << fixed << ourTPS << " & ";
        if (equi.stages.empty()) cout << "OOM"; else cout << setprecision(3) << fixed << equiTPS;
        cout << " & "
             << setprecision(3) << fixed << ratio << "$\\times$ & "
             << setprecision(1) << fixed << runtime << "s \\\\\n";
    }
}


void runGNMT () {
    vector<pair<int,double>> pairs = {
        {2, 2.5}, {2, 3.5},
        {4, 1.2}, {4, 2.5},
        {8, 0.6}, {8, 1.2}, {8, 2.4},
        {16, 0.3}, {16, 0.6}, {16, 1.2},
        {32, 0.3},
        {64, 0.3},
        {32, 0.8}
    };
    sort(pairs.begin(), pairs.end());
    for (pair<int,double> mm : pairs) {
        cerr << endl << endl << endl << endl;
        DBG(mm.first);
        DBG(mm.second);
        Instance ins = readGNMT();
        ins.maxMemoryPerDevice = mm.second * (1 << 30);
        ins.maxDevices = mm.first;
        //ins.mbsInBatch = 256;
        ins.bandwidth = 25.0 * (1LL << 30);
        
        // our alg
        clock_t startTime = clock();
        Graph g(ins);
        Result our = g.runDP();
        double runtime = (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;
        DBG(runtime)
        if (our.stages.empty()) {
            continue; // infeasible, not interesting
        }
        double ourTPS = g.getTPSForResult(our);

        Result equi = runPipeDream2BWPlannerNonTransformer(ins, false);
        double equiTPS = equi.stages.empty() ? 1e30 : g.getTPSForResult(equi);
        double ratio = equi.stages.empty() ? 0 : ourTPS / equiTPS;

        cout << mm.first << " & "
             << mm.second << " & "
             << setprecision(3) << fixed << ourTPS << " & ";
        if (equi.stages.empty()) cout << "OOM"; else cout << setprecision(3) << fixed << equiTPS;
        cout << " & "
             << setprecision(3) << fixed << ratio << "$\\times$ & "
             << setprecision(1) << fixed << runtime << "s \\\\\n";
    }
}


int main (int argc, char **argv) {
    plots(); // generates data for the bar plots (Fig. 1)

    parallelizabilityExperiment(32, 8, 960); // Fig. 2a
    parallelizabilityExperiment(32, 8, 512); // Fig. 2b

    scalability(); // Fig. 3

    correlationExperiment(); // Fig. 4 (y-axis values)

    // experiment for Appendix C
    OUTPUT_KNAPSACK_INSTANCES_FOR_INSPECTION = true;
    single();
    OUTPUT_KNAPSACK_INSTANCES_FOR_INSPECTION = false;

    // additional experiments on Resnet50 and GNMT
    runGNMT();
    cout << endl;
    runResnet();
}
