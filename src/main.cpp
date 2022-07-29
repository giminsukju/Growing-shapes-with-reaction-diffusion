#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/lscm.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/loop.h>
#include <igl/writeOBJ.h>
#include <igl/procrustes.h>
#include <igl/rotate_vectors.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/edges.h>
#include <igl/is_boundary_edge.h>
#include <igl/edge_flaps.h>
#include <igl/edge_topology.h>


#include<Eigen/SparseCholesky>	
#include <Eigen/Core>


#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"


#include <iostream>
#include <unordered_set>
#include <utility>


// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
Eigen::VectorXd original_scaling;

// Options for algorithms
int iVertexSource = 7;

int n; // # of vertices in the mesh
Eigen::VectorXd randN;

void create2DGridManual(int n, int m, Eigen::MatrixXd &V, Eigen::MatrixXi &F, float scale_1, float scale_2, bool displace, bool rotate) {
    using namespace Eigen;
    V.resize((n+1)*(m+1), 3);
    F.resize(n*m*2, 3);

    for (int i = 0; i < (n + 1); i++) { // row
        for (int j = 0; j < (m + 1); j++) { // col
            float z;
            if (displace) {
                z = j * 0.5 + (float(rand()) / float((RAND_MAX)) * float(5.0));
            }
            else {
                z = 0;
            }
            if (rotate) {
                V.row((m + 1) * i + j) = Vector3d(-j * scale_1, z, -i * scale_2);
            }
            else {
                V.row((m + 1) * i + j) = Vector3d(-j * scale_1, -i * scale_2, z);
            }
        }
    }

    int idx = 0;
    for (int i = 0; i < (n); i++) {
        for (int j = 0; j < (m); j++) {
            F.row(idx) = Vector3i((m + 1) * i + j, (m + 1) * (i + 1) + j, (m + 1) * i + j + 1);
            idx += 1;
            F.row(idx) = Vector3i((m + 1) * (i + 1) + j, (m + 1) * (i + 1) + j + 1, (m + 1) * i + j + 1);
            idx += 1;
        }
    }
}

void create2DGridVariationsAndSave() {
    Eigen::MatrixXd V1, V2, V3, V4;
    Eigen::MatrixXi F1, F2, F3, F4;
    create2DGridManual(10, 20, V1, F1, 1, 1, false, false);
    igl::writeOBJ("../original grid.obj", V1, F1);

    create2DGridManual(10, 20, V2, F2, 2, 3, false, false);
    igl::writeOBJ("../scaled grid.obj", F2, F2);

    create2DGridManual(10, 20, V3, F3, 1, 1, true, false);
    igl::writeOBJ("../z displaced grid.obj", V3, F3);

    create2DGridManual(10, 20, V4, F4, 1, 1, false, true);
    igl::writeOBJ("../y rotated grid.obj", V4, F4);
}

// Turing Semi-Implicit 2
void reactionDiffusionImplicit(float t, float _alpha, float _beta, float _s, float _da, float _db) {
    using namespace Eigen;
    using namespace std;

    SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L); // returns V by V

    VectorXd a = VectorXd::Constant(n, 1, 4); // initial a
    VectorXd b = VectorXd::Constant(n, 1, 4); // initial b

    //Sanderson et al
    VectorXd alpha = VectorXd::Constant(n, 1, 12) + randN;  // decay rate of a
    VectorXd beta = VectorXd::Constant(n, 1, 16) + randN; // growing rate of b
    float s = (float)1 / 128; // reaction rate
    float da = (float)0.14; // diffusion rate of as
    float db = (float)0.04; // diffusion raate of b
    float deltat = 0.1; // time step

    //Sliders
    //VectorXd alpha = VectorXd::Constant(n, 1, _alpha);  // decay rate of a
    //VectorXd beta = VectorXd::Constant(n, 1, _beta); // growing rate of b
    //float s = _s; // reaction rate
    //float da = _da; // diffusion rate of as
    //float db = _db; // diffusion raate of b
    //float deltat = 0.5; // time step

    SparseMatrix<double> Y(n, n), Z(n, n), b_diag(n, n), a_diag(n, n), I(n, n); I.setIdentity();

    // Turing's model
    for (int i = 0; i < t; i++) {
        for (int i = 0; i < n; i++) {
            a_diag.coeffRef(i, i) = a(i, 1);
            b_diag.coeffRef(i, i) = b(i, 1);
        }

        Y = I - (deltat * s * b_diag) + (deltat * s * I) - (deltat * da * L);
        Z = I + (deltat * s * a_diag) - (deltat * db * L);

        SimplicialLDLT<SparseMatrix<double> > solver_A, solver_B;
        solver_A.compute(Y);
        solver_B.compute(Z);

        a = solver_A.solve(a - (deltat * s * alpha));
        b = solver_B.solve(b + (deltat * s * beta));
    }

    cout << "implicit " << a(1, 1) << endl;
    cout << alpha(1, 1) << " " << beta(1, 1) << " " << s << " " << da << " " << db << endl;
    auto mesh = polyscope::getSurfaceMesh("input mesh");
    auto temp = mesh->addVertexScalarQuantity("RD-implicit", a);
    //temp->setMapRange({ 3.3,4.7 });
}

void reactionDiffusionExplicit(float t, float _alpha, float _beta, float _s, float _da, float _db) {
    using namespace Eigen;
    using namespace std;

    SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L); // returns V by V
    //igl::massmatrix(meshV, meshF, igl::MASSMATRIX_TYPE_VORONOI, M); // returns V by V
    //SimplicialLDLT<SparseMatrix<double> > solver;
    //solver.compute(M);//diagonal
    //L = solver.solve(L); // L = M^(-1)L

    VectorXd a = VectorXd::Constant(n, 1, 4) + randN;
    VectorXd b = VectorXd::Constant(n, 1, 4) + randN;

    //Sanderson et al
    VectorXd alpha = VectorXd::Constant(n, 1, 12);  // decay rate of a
    VectorXd beta = VectorXd::Constant(n, 1, 16); // growing rate of b
    float s = (float)1/128; // reaction rate
    float da = (float)1/16; // diffusion rate of as
    float db = (float)1/4; // diffusion raate of b
    float deltat = 0.1; // time step

    //Weird Zebra shape
    //VectorXd alpha = VectorXd::Constant(n, 1, 0.233);  // decay rate of a
    //VectorXd beta = VectorXd::Constant(n, 1, 0.465); // growing rate of b
    //float s = 0.058; // reaction rate
    //float da = 0.837; // diffusion rate of as
    //float db = 0.070; // diffusion raate of b
    //float deltat = 0.5; // time step

    //Sliders
    //VectorXd alpha = VectorXd::Constant(n, 1, _alpha);  // decay rate of a
    //VectorXd beta = VectorXd::Constant(n, 1, _beta); // growing rate of b
    //float s = _s; // reaction rate
    //float da = _da; // diffusion rate of as
    //float db = _db; // diffusion raate of b
    //float deltat = 0.5; // time step
   
    // Turing's model. B -> A
    for (int i = 0; i < t; i++) {
        a = (a + deltat * s * (a.cwiseProduct(b) - a - alpha) + deltat * da * L * a).eval();
        b = (b + deltat * s * (beta - a.cwiseProduct(b)) + deltat * db * L * b).eval();
    }

    cout << a(1, 1)  <<  endl;
    cout << alpha(1,1) << " " << beta(1,1) << " " << s << " " << da << " " << db << endl;
    auto mesh = polyscope::getSurfaceMesh("input mesh");
    auto temp = mesh->addVertexScalarQuantity("RD-explicit", a);
    temp -> setMapRange({3.3,4.7});
}

void computeCotangentLaplacian(float t) {
  using namespace Eigen;
  using namespace std;

  VectorXd A;
  igl::gaussian_curvature(meshV, meshF, A); // A: V by 1 
  float deltat = 0.1;
  SparseMatrix<double> L;
  igl::cotmatrix(meshV, meshF, L); // returns V by V

  // explicit 
   //A = (A + t*L*A).eval();

   //auto mesh = polyscope::getSurfaceMesh("input mesh");
   //auto temp = mesh->addVertexScalarQuantity("laplacian", A);
   //temp -> setMapRange({-0.1,0.1});

  //semi-implicit
  SparseMatrix<double> Y;

  SparseMatrix<double> I(n, n);
  I.setIdentity();

  Y = I- deltat * L;

  SimplicialLDLT<SparseMatrix<double> > solver;
  solver.compute(Y);

  for (int i = 0; i < t; i++) {
      A = solver.solve(A);
  }

  auto mesh = polyscope::getSurfaceMesh("input mesh");
  auto temp = mesh->addVertexScalarQuantity("laplacian", A);
  temp -> setMapRange({-0.1, 0.1});
}

// gray-scott semi-implicit (1 or 2?)
void implicitReactionDiffusionGrayScott(float numSteps, float timeStep, double F_val, double k_val) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(meshV, meshF, L);

    Eigen::VectorXd U = Eigen::VectorXd::Constant(L.rows(), 1, 1);
    Eigen::VectorXd V = Eigen::VectorXd::Constant(L.rows(), 1, 0);

    Eigen::VectorXd F = Eigen::VectorXd::Constant(L.rows(), 1, F_val);  // feed rate
    Eigen::VectorXd k = Eigen::VectorXd::Constant(L.rows(), 1, k_val); // degrading rate
    float du = 1; // diffusion rate
    float dv = (float)0.5; // diffusion rate

    // Modify V partially (eg ink drop in water) to drive diffusion
    int reference_idx = rand() % (meshV.rows()+1); //random index between 0 and # rows in meshV (ie # of vertices)
    // float desired_dist = 0.1; // for spot (about 1/20 of its size)
    float desired_dist = 1; // for grid (about 1/20 of its size)
    for (int i = 0; i < meshV.rows(); i++) {
        // when the Euclidean distance bw the reference idx vertex position & ith vertex < 0.1
        if ((pow(( meshV(i, 0) - meshV(reference_idx, 0)),2) + \
            pow((meshV(i, 1) - meshV(reference_idx, 1)),2)  + \
            pow((meshV(i, 2) - meshV(reference_idx, 2)), 2))  < pow(desired_dist, 2)) {
            V(i) = 1.0;
        }
    }

    Eigen::SparseMatrix<double> I(L.rows(), L.rows()); I.setIdentity();

    Eigen::SparseMatrix<double> temp1 = (I - timeStep * du * L).eval();
    Eigen::SparseMatrix<double> temp2 = (I - timeStep * dv * L).eval();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver1;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver2;

    solver1.compute(temp1);
    solver2.compute(temp2);

    for (int i = 0; i < numSteps; i++) {
        Eigen::VectorXd UVV = U.array() * V.array() * V.array();
        Eigen::VectorXd FU = F.array() * U.array();
        U = solver1.solve((U + timeStep * (F - FU - UVV)).eval());
        Eigen::VectorXd kV = k.array() * V.array();
        Eigen::VectorXd FV = F.array() * V.array();
        V = solver2.solve((V + timeStep * (UVV - kV - FV)).eval());
    }

    original_scaling = U; //update scaling as a global variable
    auto temp = polyscope::getSurfaceMesh("input mesh");
    auto mesh = temp->addVertexScalarQuantity("Scott Implicit", U);
}

void procruste(Eigen::MatrixXi F_init, Eigen::MatrixXd V_init, Eigen::MatrixXd V_rest, Eigen::MatrixXd& V_update) {
    int n_faces = F_init.rows();
    for (int i = 0; i < n_faces; i++) { // iterating triangle by triangle 
        Eigen::VectorXi v_indices = F_init.row(i); //vertex indices in this triangle

        Eigen::MatrixXd V_init_triangle, V_rest_triangle;
        igl::slice(V_init, v_indices, 1, V_init_triangle); //slice: Y = X(I,:)
        igl::slice(V_rest, Eigen::Vector3i(3*i, 3*i+1, 3*i+2), 1, V_rest_triangle); //slice: Y = X(I,:)

        double scale;
        Eigen::MatrixXd R;
        Eigen::VectorXd t;
        igl::procrustes(V_rest_triangle, V_init_triangle, false, false, scale, R, t); //X: reference, Y:target
        Eigen::MatrixXd V_procruste_triangle = (V_rest_triangle * R).rowwise() + t.transpose();
        igl::slice_into(V_procruste_triangle, v_indices, 1, V_init); //slice into: Y(I,:) = X
    }
    V_update = V_init;
}

// TODO: combine with procruste triangle
void procruste_diamond(Eigen::MatrixXi F_rest_diamonds_in_org_v_idx, Eigen::MatrixXd V_init, Eigen::MatrixXd V_rest_diamonds, Eigen::MatrixXd& V_update) {
    // rigidly translate & rotate a flat diamond to match the initial mesh as much as possible
    int n_diamonds = F_rest_diamonds_in_org_v_idx.rows();
    for (int i = 0; i < n_diamonds; i++) { // iterating triangle by triangle 
        Eigen::MatrixXd V_init_diamond, V_rest_diamond;

        Eigen::VectorXi v_indices = F_rest_diamonds_in_org_v_idx.row(i); //vertex indices from the upper triangle in this diamond
        igl::slice(V_init, v_indices, 1, V_init_diamond); //slice: Y = X(I,:), 4 by 3, 
        igl::slice(V_rest_diamonds, Eigen::Vector4i(4 * i, 4 * i + 2, 4 * i + 1, 4 * i + 3), 1, V_rest_diamond); //slice: Y = X(I,:), 4 by 3

        double scale;
        Eigen::MatrixXd R;
        Eigen::VectorXd t;
        igl::procrustes(V_rest_diamond, V_init_diamond, false, false, scale, R, t); //X: reference, Y:target
        Eigen::MatrixXd V_procruste_diamond = (V_rest_diamond * R).rowwise() + t.transpose(); // 2 by 3
        igl::slice_into(V_procruste_diamond, v_indices, 1, V_init); //slice into: Y(I,:) = X
    }
    V_update = V_init;
}

//TODO: check if this always create valid triangles (eg when one vertes is scaled by 0.1 and the other is by 2.0)
void constructRestMesh(Eigen::MatrixXd V_init, Eigen::MatrixXi F_init, Eigen::VectorXd scaling, Eigen::MatrixXd& V_rest, Eigen::MatrixXi& F_rest) {
    int n_faces = F_init.rows();
    for (int i = 0; i < n_faces; i++) {
        Eigen::VectorXi v_indices = F_init.row(i); //vertex indices in this triangle, 1 by 3

        Eigen::VectorXd v0 = V_init.row(v_indices(0)); //original position of the 1st vertex in the triangle, 1 by 3 //TODO: randomly select the 1st vertex
        Eigen::VectorXd v1 = V_init.row(v_indices(1)); //original position of the 2nd vertex in the triangle, 1 by 3
        Eigen::VectorXd v2 = V_init.row(v_indices(2)); //original position of the 3rd vertex in the triangle, 1 by 3

        float new_v0_v1_l = (v1.transpose() - v0.transpose()).norm() * (scaling(v_indices(0)) + scaling(v_indices(1))) / 2;
        float new_v1_v2_l = (v2.transpose() - v1.transpose()).norm() * (scaling(v_indices(1)) + scaling(v_indices(2))) / 2;
        float new_v2_v0_l = (v0.transpose() - v2.transpose()).norm() * (scaling(v_indices(2)) + scaling(v_indices(0))) / 2;

        float new_v2_x = (pow(new_v0_v1_l, 2) + pow(new_v2_v0_l, 2) - pow(new_v1_v2_l, 2)) / (2 * new_v0_v1_l);
        float new_v2_y = sqrt(pow(new_v2_v0_l, 2) - pow(new_v2_x, 2));

        V_rest.row(i * 3) = Eigen::Vector3d(0.0, 0.0, 0.0);
        V_rest.row(i * 3 + 1) = Eigen::Vector3d(new_v0_v1_l, 0.0, 0.0);
        V_rest.row(i * 3 + 2) = Eigen::Vector3d(new_v2_x, new_v2_y, 0.0);

        F_rest.row(i) = Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2);
    }
}

void constructDiamondRestMesh(Eigen::MatrixXd V_init, Eigen::MatrixXi F_init, Eigen::MatrixXd& V_rest_diamond, Eigen::MatrixXi& F_rest_diamond, Eigen::MatrixXi& F_rest_diamond_in_org_v_idx) {
    Eigen::MatrixXi E; //E(e) row = (vertex i, vertex j) edge
    Eigen::VectorXi EMAP;
    Eigen::MatrixXi EF; //E(e) is the edge of EF(e,0) face and EF(e,1) face, edge having EF(e,0) == -1 or EF(e,1) == -1 indicates boundary edge
    Eigen::MatrixXi EI; //E(e,0) is opposite of EI(e,0)=v th vertex in that face (and similarly for E(e,1))
    igl::edge_flaps(F_init, E, EMAP, EF, EI);

    int inner_edge_ctr = 0;
    for (int i = 0; i < E.rows(); i++) { // iterate over all inner edges
        int face_1_idx = EF(i, 0);
        int face_2_idx = EF(i, 1);

        int edge_v_1_idx = E(i, 0);
        int edge_v_2_idx = E(i, 1);

        int face_1_corner_v_idx = F_init(face_1_idx, EI(i, 0));
        int face_2_corner_v_idx = F_init(face_2_idx, EI(i, 1));
        if ((face_1_idx != -1) && (face_2_idx != -1)) { // is inner edge
            // construct a flattened diamond that loks like:  
            //              
            //           edge_v_1_idx (v2)
            //           /             \
            //          /               \
            //  edge_v_1_idx (v0) ------ edge_v_2_idx (v1)
            //          \               /
            //           \             /
            //            edge_v_2_idx (v3)
            //
            Eigen::VectorXd v0 = V_init.row(edge_v_1_idx);
            Eigen::VectorXd v1 = V_init.row(edge_v_2_idx);
            Eigen::VectorXd v2 = V_init.row(face_1_corner_v_idx);
            Eigen::VectorXd v3 = V_init.row(face_2_corner_v_idx);

            float v0_v1_l = (v1.transpose() - v0.transpose()).norm();
            float v1_v2_l = (v2.transpose() - v1.transpose()).norm();
            float v0_v2_l = (v0.transpose() - v2.transpose()).norm();
            float v1_v3_l = (v1.transpose() - v3.transpose()).norm();
            float v0_v3_l = (v0.transpose() - v3.transpose()).norm();

            float flat_v1_x = v0_v1_l;
            float flat_v1_y = 0;
            float flat_v2_x = (pow(v0_v1_l, 2) + pow(v0_v2_l, 2) - pow(v1_v2_l, 2)) / (2 * v0_v1_l);//TODO: cleanup and wrap as a function
            float flat_v2_y = sqrt(pow(v0_v2_l, 2) - pow(flat_v2_x, 2));
            float flat_v3_x = (pow(v0_v1_l, 2) + pow(v0_v3_l, 2) - pow(v1_v3_l, 2)) / (2 * v0_v1_l);
            float flat_v3_y = -sqrt(pow(v0_v3_l, 2) - pow(flat_v3_x, 2));

            V_rest_diamond.row(inner_edge_ctr * 4) = Eigen::Vector3d(0.0, 0.0, 0.0);
            V_rest_diamond.row(inner_edge_ctr * 4 + 1) = Eigen::Vector3d(flat_v1_x, flat_v1_y, 0.0);
            V_rest_diamond.row(inner_edge_ctr * 4 + 2) = Eigen::Vector3d(flat_v2_x, flat_v2_y, 0.0);
            V_rest_diamond.row(inner_edge_ctr * 4 + 3) = Eigen::Vector3d(flat_v3_x, flat_v3_y, 0.0);

            F_rest_diamond.row(inner_edge_ctr) = Eigen::Vector4i(inner_edge_ctr * 4, inner_edge_ctr * 4 + 2, inner_edge_ctr * 4 + 1, inner_edge_ctr * 4 + 3); // v0 -> v2 -> v1 -> v3 order
            F_rest_diamond_in_org_v_idx.row(inner_edge_ctr) = Eigen::Vector4i(edge_v_1_idx, face_1_corner_v_idx, edge_v_2_idx, face_2_corner_v_idx); // v0 -> v2 -> v1 -> v3 order
            //if (inner_edge_ctr == 200) { //TODO: debugging purposes, have to update declarations
            //    Eigen::MatrixXd V_rest_diamond_1(1 * 4, 3);
            //    Eigen::MatrixXi F_rest_diamond_1(1 * 2, 3);
            //    V_rest_diamond_1.row(0) = Eigen::Vector3d(0.0, 0.0, 0.0);
            //    V_rest_diamond_1.row(1) = Eigen::Vector3d(flat_v1_x, flat_v1_y, 0.0);
            //    V_rest_diamond_1.row(2) = Eigen::Vector3d(flat_v2_x, flat_v2_y, 0.0);
            //    V_rest_diamond_1.row(3) = Eigen::Vector3d(flat_v3_x, flat_v3_y, 0.0);

            //    F_rest_diamond_1.row(0) = Eigen::Vector4i(0, 2, 1, 3);

            //    polyscope::registerSurfaceMesh("diamond rest mesh edge 200", V_rest_diamond_1, F_rest_diamond_1);
            //}
            inner_edge_ctr += 1;
        }
    }
}

void growShape(int n_iteration, double alpha, double beta, double min_scaling, double max_scaling) { //TODO: take mesh file as an input
    std::cout << n_iteration << " " << alpha << "  " << beta << " " << min_scaling << " " << max_scaling << std::endl;

    Eigen::MatrixXd V_init = meshV;
    Eigen::MatrixXi F_init = meshF;

    // set non-uniform scaling, inner area scaling > outer area scaling
    //Eigen::VectorXd scaling = Eigen::VectorXd::Constant(V_init.rows(), 1, 1.2);
    //int m = 10; //TODO: take input mesh size as a variable
    //for (int i = 5; i <= 15; i++) { // row
    //    for (int j = 2; j <= 8; j++) { // col
    //        scaling((m + 1) * i + j) = 1.5;
    //    }
    //}

    // set non-uniform scaling, spot face scaling > spot body scaling
    //Eigen::VectorXd scaling = Eigen::VectorXd::Constant(V_init.rows(), 1, 1.2);
    //Eigen::VectorXd reference = V_init.row(64); //roughly the farthest point from spot's body, spot's head starting point
    //float head_dist = 0.66; //roughly Euclidean distance bw the reference point and the boundary of spot's body and head
    //for (int i = 0; i < V_init.rows(); i++) { 
    //    Eigen::VectorXd v = V_init.row(i);
    //    float dist = (reference.transpose() - v.transpose()).norm();
    //    if (dist <= head_dist) {
    //        scaling(i) = 1.5;
    //    }
    //}

    Eigen::VectorXd scaling  = ((max_scaling - min_scaling) * original_scaling) + Eigen::VectorXd::Constant(V_init.rows(), 1, min_scaling);

    // construct rest mesh
    int n_faces = F_init.rows();
    Eigen::MatrixXd V_rest(n_faces * 3, 3); // position of n_faces * 3 vertices, ordered according to F_rest face
    Eigen::MatrixXi F_rest(n_faces, 3);
    constructRestMesh(V_init, F_init, scaling, V_rest, F_rest);
    //polyscope::registerSurfaceMesh("triangle rest mesh", V_rest, F_rest);

    for (int itr_ctr = 0; itr_ctr < n_iteration; itr_ctr++) {
        // stretching 
        Eigen::MatrixXd V_stretched(V_init.rows(), 3);
        procruste(F_init, V_init, V_rest, V_stretched); //update V_stretched mesh
        V_init = V_init + (V_stretched - V_init) * alpha;
        //if (itr_ctr == 0) { //debugging purposes, TODO: use f-string instead of multiple if statements
        //    polyscope::registerSurfaceMesh("stretched: iter 1", V_init, F_init);
        //}
        // construct diamond rest mesh for bending-resisting
        Eigen::MatrixXi E; //E(e) row = (vertex i, vertex j) edge
        Eigen::VectorXi EMAP;
        Eigen::MatrixXi EF; //E(e) is the edge of EF(e,0) face and EF(e,1) face, edge having EF(e,0) == -1 or EF(e,1) == -1 indicates boundary edge
        Eigen::MatrixXi EI; //E(e,0) is opposite of EI(e,0)=v th vertex in that face (and similarly for E(e,1))
        igl::edge_flaps(F_init, E, EMAP, EF, EI); //TODO: replace with edges()

        Eigen::Array<bool, Eigen::Dynamic, 1> BoolBoundary;
        igl::is_boundary_edge(E, F_init, BoolBoundary); //TODO: remove this function, redundant
        int n_inner_edges = E.rows() - BoolBoundary.count();

        Eigen::MatrixXd V_rest_diamond(n_inner_edges * 4, 3); // xyz position of n_inner_edges * 4 vertices, (4*i)th, (4*i+1)th, (4*i+2)th, (4*i+3)th vertices make one diamond, diamond ordered according to inner edge list
        Eigen::MatrixXi F_rest_diamond(n_inner_edges, 4); // vertex indices of a diamond consist of (2*i)th triangle and (2*i+1)th triangl
        Eigen::MatrixXi F_rest_diamond_in_org_v_idx(n_inner_edges, 4); // vertex indices of a diamond consist of (2*i)th triangle and (2*i+1)th triangl
        constructDiamondRestMesh(V_init, F_init, V_rest_diamond, F_rest_diamond, F_rest_diamond_in_org_v_idx);
        //if (itr_ctr == 0) {
        //    polyscope::registerSurfaceMesh("diamond rest mesh: iter 1", V_rest_diamond, F_rest_diamond);
        //}

        //bending resisting
        Eigen::MatrixXd V_bending_resisted(V_init.rows(), 3);
        procruste_diamond(F_rest_diamond_in_org_v_idx, V_init, V_rest_diamond, V_bending_resisted); //update V_bending_resisted mesh
        V_init = V_init + (V_bending_resisted - V_init) * beta;
        //if (itr_ctr == 0) { //debugging purposes, TODO: use f-string instead of multiple if statements
        //    polyscope::registerSurfaceMesh("bending resisted: iter 1", V_init, F_init);
        //}
    }
    polyscope::registerSurfaceMesh("final mesh", V_init, F_init);
}

void callback() {
    static float timestep = 1;
    ImGui::InputFloat("Time Step", &timestep);

    static double F = 0.024;
    ImGui::InputDouble("F", &F);

    static double k = 0.06;
    ImGui::InputDouble("k", &k);

    static float rd_n_iteration = 1;
    ImGui::InputFloat("Number of Steps", &rd_n_iteration);

    if (ImGui::Button("Gray-Scott Reaction Diffusion Implicit")) {
        implicitReactionDiffusionGrayScott(rd_n_iteration, timestep, F, k); // scaling range: (0, 1)
    }

    static double min_scaling = 1; //scaling results from r-d would be scaled accordingly
    ImGui::InputDouble("Minimum scaling", &min_scaling);

    static double max_scaling = 2.5; //scaling results from r-d would be scaled accordingly
    ImGui::InputDouble("Maximum scaling", &max_scaling);

    static double alpha = 1.0;
    ImGui::InputDouble("alpha", &alpha);

    static double beta = 0.5;
    ImGui::InputDouble("beta", &beta);

    //static int growing_max_n_iteration = 100;
    //ImGui::InputInt("max iter", &growing_max_n_iteration);

    static int growing_n_iteration = 50;
    ImGui::InputInt("Growing iterations", &growing_n_iteration);

    if (ImGui::Button("growing shapes refresh")) {
        growShape(growing_n_iteration, alpha, beta, min_scaling, max_scaling);
    }


    //static int growing_n_iteration = 0;
    //if ((ImGui::SliderInt("growing shapes", &growing_n_iteration, 0, growing_max_n_iteration)) ||
    //    (ImGui::Button("growing shapes refresh"))) {
    //    growShape(growing_n_iteration, alpha, beta, min_scaling, max_scaling);
    //}
}


int main(int argc, char **argv) {
    polyscope::init();

    //std::string filename = "../spot.obj";
    std::string filename = "../original grid.obj";

    Eigen::MatrixXd origV, meshV1;
    Eigen::MatrixXi origF, meshF1;
    igl::readOBJ(filename, origV, origF);
    if (filename == "../original grid.obj") {
        for (int i = 0; i < origV.rows(); i++) {
            origV(i, 2) = (float(rand()) / float((RAND_MAX)) * float(0.1));
        }
    }

    Eigen::SparseMatrix<double> S, S1;
    igl::loop(origV.rows(), origF, S1, meshF1);
    meshV1 = S1 * origV;
    igl::loop(meshV1.rows(), meshF1, S, meshF);
    meshV = S * meshV1;

    polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

    polyscope::state::userCallback = callback;

    polyscope::show();

    return 0;
}
