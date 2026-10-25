#ifndef HDBSCAN_H
#define HDBSCAN_H

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <limits>
#include <numeric>
#include <unordered_map>

using namespace Eigen;
using namespace std;

class HDBSCAN {
public:
    // ------------------------------------------------------------------
    // DBSCAN — fixed-radius, fixed-density clustering. Unchanged from the
    // original file. Kept around as a cheap baseline / building block:
    // findNeighbors() below is reused nowhere by HDBSCAN* (core distance
    // needs k-NN distance, not an eps-radius query), but the method itself
    // is still useful on its own for a quick single-density pass.
    // ------------------------------------------------------------------
    vector<int> cluster(const MatrixXd& data, double eps, int minPts) {
        int n = data.rows();
        vector<int> labels(n, -1);  // -1 means unvisited, 0 means noise
        int cluster_id = 0;

        for (int i = 0; i < n; i++) {
            if (labels[i] != -1) continue;  // Already processed

            vector<int> neighbors = findNeighbors(data, i, eps);

            if ((int)neighbors.size() < minPts) {
                labels[i] = 0;  // Mark as noise
                continue;
            }

            labels[i] = cluster_id;
            queue<int> seeds;
            for (int neighbor : neighbors) {
                seeds.push(neighbor);
            }

            while (!seeds.empty()) {
                int current_point = seeds.front();
                seeds.pop();

                if (labels[current_point] == 0) {
                    labels[current_point] = cluster_id;  // Change noise to cluster
                }

                if (labels[current_point] == -1) {
                    labels[current_point] = cluster_id;

                    vector<int> current_neighbors = findNeighbors(data, current_point, eps);
                    if ((int)current_neighbors.size() >= minPts) {
                        for (int neighbor : current_neighbors) {
                            if (labels[neighbor] == -1) {
                                seeds.push(neighbor);
                            }
                        }
                    }
                }
            }

            cluster_id++;
        }

        return labels;
    }

    // ------------------------------------------------------------------
    // True HDBSCAN* (Campello, Moulavi & Sander, 2013): clusters across a
    // *range* of densities instead of DBSCAN's single eps, then keeps
    // whichever clusters are the most statistically persistent ("excess of
    // mass"). No eps parameter — that's the point of HDBSCAN over DBSCAN.
    //
    //   minSamples     neighborhood size used to estimate each point's local
    //                  density (its "core distance" — distance to its
    //                  minSamples-th nearest neighbor).
    //   minClusterSize smallest group of points HDBSCAN is willing to call
    //                  its own cluster; smaller groups get folded into a
    //                  parent cluster or, if nothing containing them
    //                  survives selection, marked as noise.
    //
    // Returns one label per row of `data`: -1 = noise, 0..k-1 = cluster id.
    // Call membershipStrength() afterward for a per-point confidence score
    // in [0,1] (0 for noise) — how deep into its cluster a point sits.
    //
    // Complexity is O(n^2) (brute-force core distances + Prim's MST), same
    // order as the rest of this codebase's dense-matrix approach (see
    // Clustering.h's similarityMatrix). Fine at the sizes used here; a
    // large survey would want a spatial index / dual-tree Boruvka instead.
    // ------------------------------------------------------------------
    vector<int> clusterAdvanced(const MatrixXd& data, int minSamples = 5, int minClusterSize = 5) {
        int n = data.rows();
        if (n == 0) { membershipStrength_.clear(); return {}; }
        if (n < minClusterSize) {
            // Not enough points to ever form a cluster of the requested size.
            membershipStrength_.assign(n, 0.0);
            return vector<int>(n, -1);
        }

        vector<double> coreDist = coreDistances(data, minSamples);
        vector<MSTEdge> mst = buildMST(data, coreDist);
        vector<LinkageNode> tree = buildLinkageTree(mst, n);

        vector<CondensedEdge> edges;
        unordered_map<int, double> birthLambda;
        int numClusters = 0;
        condenseTree(tree, n, minClusterSize, edges, birthLambda, numClusters);

        vector<double> strength;
        vector<int> labels = extractFlatClusters(edges, birthLambda, n, numClusters, strength);
        membershipStrength_ = strength;
        return labels;
    }

    const vector<double>& membershipStrength() const { return membershipStrength_; }

private:
    // ---- DBSCAN helper (unchanged) ----
    vector<int> findNeighbors(const MatrixXd& data, int point_idx, double eps) {
        vector<int> neighbors;
        RowVectorXd point = data.row(point_idx);

        for (int i = 0; i < data.rows(); i++) {
            if (i == point_idx) continue;
            double distance = (point - data.row(i)).norm();
            if (distance <= eps) {
                neighbors.push_back(i);
            }
        }

        return neighbors;
    }

    // ==================================================================
    // HDBSCAN* internals
    // ==================================================================

    struct MSTEdge { int a, b; double weight; };
    struct LinkageNode { int left, right; double dist; int size; };
    // A row of the condensed tree: `child_size` points transition out of
    // `parent` (into either a brand-new child cluster, if child >= n and
    // child_size >= minClusterSize, or as individual noise-candidates,
    // if child < n and child_size == 1) at density level lambda_val.
    struct CondensedEdge { int parent; int child; double lambda_val; int child_size; };

    struct UnionFind {
        vector<int> parent, sz;
        explicit UnionFind(int n) : parent(n), sz(n, 1) { iota(parent.begin(), parent.end(), 0); }
        int find(int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
        void unite(int a, int b) {
            a = find(a); b = find(b);
            if (a == b) return;
            if (sz[a] < sz[b]) swap(a, b);
            parent[b] = a;
            sz[a] += sz[b];
        }
    };

    // Step 1: core distance of each point = distance to its minSamples-th
    // nearest neighbor (a local density estimate — smaller means denser).
    vector<double> coreDistances(const MatrixXd& data, int minSamples) {
        int n = data.rows();
        vector<double> core(n);

		#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            vector<double> d;
            d.reserve(n - 1);
            RowVectorXd pi = data.row(i);
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                d.push_back((pi - data.row(j)).norm());
            }
            int k = min((int)d.size(), minSamples) - 1;
			k = max(k, 0);
			nth_element(d.begin(), d.begin() + k, d.end()); // only need the k-th smallest, not a full sort
			core[i] = d[k];
        }
        return core;
    }

    // Step 2: minimum spanning tree over the *mutual reachability* graph,
    // where d_mreach(a,b) = max(core[a], core[b], euclidean(a,b)). This is
    // what makes HDBSCAN robust to a single fixed eps: sparse regions get
    // effectively pushed apart by their large core distances.
    vector<MSTEdge> buildMST(const MatrixXd& data, const vector<double>& coreDist) {
        int n = data.rows();
        vector<bool> inMST(n, false);
        vector<double> minEdge(n, numeric_limits<double>::infinity());
        vector<int> parent(n, -1);
        minEdge[0] = 0.0;
        vector<MSTEdge> mst;
        mst.reserve(n - 1);

        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = numeric_limits<double>::infinity();
            for (int v = 0; v < n; v++) {
                if (!inMST[v] && minEdge[v] < best) { best = minEdge[v]; u = v; }
            }
            inMST[u] = true;
            if (parent[u] != -1) mst.push_back({parent[u], u, minEdge[u]});

            RowVectorXd pu = data.row(u);
            for (int v = 0; v < n; v++) {
                if (inMST[v]) continue;
                double eucl = (pu - data.row(v)).norm();
                double mreach = max({coreDist[u], coreDist[v], eucl});
                if (mreach < minEdge[v]) { minEdge[v] = mreach; parent[v] = u; }
            }
        }
        return mst;
    }

    // Step 3: turn the MST into a single-linkage hierarchy — merge the
    // cheapest edges first via union-find, giving each merge a fresh node
    // id (n, n+1, ... 2n-2). This is the full dendrogram, before pruning.
    vector<LinkageNode> buildLinkageTree(vector<MSTEdge> mst, int n) {
        sort(mst.begin(), mst.end(), [](const MSTEdge& a, const MSTEdge& b) { return a.weight < b.weight; });
        UnionFind uf(n);
        vector<int> compLabel(n);
        iota(compLabel.begin(), compLabel.end(), 0);
        vector<LinkageNode> tree;
        tree.reserve(n > 0 ? n - 1 : 0);
        int nextId = n;
        for (auto& e : mst) {
            int ra = uf.find(e.a), rb = uf.find(e.b);
            int la = compLabel[ra], lb = compLabel[rb];
            int sizeA = uf.sz[ra], sizeB = uf.sz[rb];
            tree.push_back({la, lb, e.weight, sizeA + sizeB});
            uf.unite(ra, rb);
            compLabel[uf.find(ra)] = nextId++;
        }
        return tree;
    }

    // All original point ids under a linkage-tree node, memoized.
    vector<int> subtreeLeaves(int nodeId, const vector<LinkageNode>& tree, int n, vector<vector<int>>& memo) {
        if (nodeId < n) return {nodeId};
        vector<int>& cached = memo[nodeId - n];
        if (!cached.empty()) return cached;
        const auto& node = tree[nodeId - n];
        vector<int> res = subtreeLeaves(node.left, tree, n, memo);
        vector<int> r = subtreeLeaves(node.right, tree, n, memo);
        res.insert(res.end(), r.begin(), r.end());
        cached = res;
        return cached;
    }

    // Step 4: condense the full dendrogram using minClusterSize. Walking
    // from the root down (largest merge distance -> smallest), whichever
    // side of a split is smaller than minClusterSize gets "exploded" into
    // individual points that leave the cluster at that density level
    // (lambda = 1/distance); the surviving side just continues under the
    // same cluster label. Only when *both* sides clear minClusterSize does
    // a real new pair of child clusters get born.
    void condenseTree(const vector<LinkageNode>& tree, int n, int minClusterSize,
                       vector<CondensedEdge>& edges, unordered_map<int, double>& birthLambda, int& numClusters) {
        int root = n + (int)tree.size() - 1;
        vector<int> relabel(root + 1, -1);
        if (root < n) { numClusters = 1; birthLambda[n] = 0.0; return; } // 0 or 1 points, nothing to merge
        relabel[root] = n;
        birthLambda[n] = 0.0;
        int nextLabel = n + 1;
        vector<vector<int>> memo(tree.size());

        for (int nodeId = root; nodeId >= n; nodeId--) {
            if (relabel[nodeId] == -1) continue; // already exploded into points by its parent
            const auto& node = tree[nodeId - n];
            int left = node.left, right = node.right;
            int leftSize = (left < n) ? 1 : tree[left - n].size;
            int rightSize = (right < n) ? 1 : tree[right - n].size;
            double lambda_val = (node.dist > 0) ? 1.0 / node.dist : numeric_limits<double>::infinity();
            int parentLabel = relabel[nodeId];
            bool leftBig = leftSize >= minClusterSize;
            bool rightBig = rightSize >= minClusterSize;

            if (leftBig && rightBig) {
                int ll = nextLabel++;
                relabel[left] = ll;
                birthLambda[ll] = lambda_val;
                edges.push_back({parentLabel, ll, lambda_val, leftSize});

                int rl = nextLabel++;
                relabel[right] = rl;
                birthLambda[rl] = lambda_val;
                edges.push_back({parentLabel, rl, lambda_val, rightSize});
            } else if (!leftBig && !rightBig) {
                for (int p : subtreeLeaves(left, tree, n, memo)) edges.push_back({parentLabel, p, lambda_val, 1});
                for (int p : subtreeLeaves(right, tree, n, memo)) edges.push_back({parentLabel, p, lambda_val, 1});
            } else if (!leftBig) {
                for (int p : subtreeLeaves(left, tree, n, memo)) edges.push_back({parentLabel, p, lambda_val, 1});
                relabel[right] = parentLabel;
            } else {
                for (int p : subtreeLeaves(right, tree, n, memo)) edges.push_back({parentLabel, p, lambda_val, 1});
                relabel[left] = parentLabel;
            }
        }
        numClusters = nextLabel - n;
    }

    // Step 5: stability of a cluster = sum over every point/child-cluster
    // that ever left it of child_size * (lambda_at_departure - lambda_birth).
    // Step 6: excess-of-mass selection — bottom-up, keep a cluster if its
    // own stability beats the combined stability of its selected
    // descendants, otherwise defer to the descendants. Finally, walk every
    // point up to the nearest selected ancestor (or noise, if none).
    vector<int> extractFlatClusters(const vector<CondensedEdge>& edges, unordered_map<int, double>& birthLambda,
                                     int n, int numClusters, vector<double>& strength) {
        int base = n;
        vector<double> stability(numClusters, 0.0);
        vector<vector<int>> childClusters(numClusters);
        vector<int> parentOfCluster(numClusters, -1);

        for (const auto& e : edges) {
            int p = e.parent - base;
            stability[p] += e.child_size * (e.lambda_val - birthLambda[e.parent]);
            if (e.child >= base) {
                childClusters[p].push_back(e.child - base);
                parentOfCluster[e.child - base] = p;
            }
        }

        vector<double> subtreeStability = stability;
        vector<bool> isCluster(numClusters, true);
        for (int c = numClusters - 1; c >= 0; c--) {
            if (childClusters[c].empty()) continue; // never split further -> keep as-is
            double childSum = 0.0;
            for (int ch : childClusters[c]) childSum += subtreeStability[ch];
            if (childSum >= stability[c]) {
                subtreeStability[c] = childSum;
                isCluster[c] = false;
            } else {
                subtreeStability[c] = stability[c];
                vector<int> stack = childClusters[c];
                while (!stack.empty()) {
                    int x = stack.back(); stack.pop_back();
                    isCluster[x] = false;
                    for (int gc : childClusters[x]) stack.push_back(gc);
                }
            }
        }

        unordered_map<int, int> outId;
        int next = 0;
        for (int c = 0; c < numClusters; c++) if (isCluster[c]) outId[c] = next++;

        vector<int> selectedAncestor(numClusters, -1);
        for (int c = 0; c < numClusters; c++) {
            int cur = c;
            while (cur != -1 && !isCluster[cur]) cur = parentOfCluster[cur];
            selectedAncestor[c] = cur;
        }

        vector<int> labels(n, -1);
        vector<double> falloutLambda(n, 0.0);
        for (const auto& e : edges) {
            if (e.child < base) {
                int p = e.parent - base;
                int anc = selectedAncestor[p];
                if (anc != -1) {
                    labels[e.child] = outId[anc];
                    falloutLambda[e.child] = e.lambda_val;
                }
            }
        }

        vector<double> maxLambda(next, 0.0);
        for (int i = 0; i < n; i++) {
            if (labels[i] != -1) maxLambda[labels[i]] = max(maxLambda[labels[i]], falloutLambda[i]);
        }

        strength.assign(n, 0.0);
        for (int i = 0; i < n; i++) {
            if (labels[i] == -1) continue;
            double m = maxLambda[labels[i]];
            strength[i] = (m > 0) ? min(1.0, falloutLambda[i] / m) : 1.0;
        }

        return labels;
    }

    vector<double> membershipStrength_;
};

#endif // HDBSCAN_H
