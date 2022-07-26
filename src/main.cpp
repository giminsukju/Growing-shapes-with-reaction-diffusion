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

// Options for algorithms
int iVertexSource = 7;

int n; // # of vertices in the mesh
Eigen::VectorXd randN;

//arbitrary function to display 2D grid color map
void function(int n, int m, Eigen::MatrixXd& func) {
    for (int i = 0; i < (n); i++) {
        for (int j = 0; j < (m); j++) {
            func(i, j) = (m + 1) * i + j;
        }
    }
}


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

// Semi-Implicit 2
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

void callback() {


  static int maxRD = 100;
  ImGui::InputInt("max iter", &maxRD);
  static float tRD = 0;
  static float alpha = 0;
  static float beta = 0;
  static float s = 0;
  static float da = 0;
  static float db = 0;

  ImGui::SameLine();
  if ((ImGui::SliderFloat("ReactionDiffusion iter", &tRD, 0, maxRD))\
      || (ImGui::SliderFloat("alpha", &alpha, 0, 20))\
      || (ImGui::SliderFloat("beta", &beta, 0, 20))\
      || (ImGui::SliderFloat("s", &s, 0, 1))\
      || (ImGui::SliderFloat("da", &da, 0, 1))\
      || (ImGui::SliderFloat("db", &db, 0, 1))){
      //reactionDiffusionExplicit(tRD, alpha, beta, s, da, db);
      reactionDiffusionImplicit(tRD, alpha, beta, s, da, db);

  }

  ImGui::SameLine();
  ImGui::InputInt("source vertex", &iVertexSource);

  ImGui::PopItemWidth();
}



void procruste(Eigen::MatrixXi F_rest, Eigen::MatrixXd V_rest, Eigen::MatrixXd V_target, Eigen::MatrixXd& V_procruste) {
    int n_faces = F_rest.rows();
    for (int i = 0; i < n_faces; i++) { // iterating triangle by triangle 
        Eigen::VectorXi v_indices = F_rest.row(i); //vertex indices in this triangle

        Eigen::MatrixXd V_rest_triangle, V_target_triangle;
        igl::slice(V_rest, v_indices, 1, V_rest_triangle); //slice: Y = X(I,:)
        igl::slice(V_target, v_indices, 1, V_target_triangle); //slice: Y = X(I,:)

        double scale;
        Eigen::MatrixXd R;
        Eigen::VectorXd t;
        igl::procrustes(V_rest_triangle, V_target_triangle, true, false, scale, R, t);
        R *= scale;
        Eigen::MatrixXd V_procruste_triangle = (V_rest_triangle * R).rowwise() + t.transpose();
        igl::slice_into(V_procruste_triangle, v_indices, 1, V_procruste); //slice into: Y(I,:) = X
    }
}

void procrusteIteration(int maxIteration) {
    Eigen::MatrixXd V_rest, V_target;
    Eigen::MatrixXi F_rest, F_target;
    // load original mesh
    igl::readOBJ("../original grid.obj", V_rest, F_rest);

    // load target mesh
    // igl::readOBJ("../scaled grid.obj", V_target, F_target);
    igl::readOBJ("../z displaced grid.obj", V_target, F_target);
    // igl::readOBJ("../y rotated grid.obj", V_target, F_target);

    polyscope::registerSurfaceMesh("rest mesh", V_rest, F_rest);
    polyscope::registerSurfaceMesh("target mesh", V_target, F_rest);

    for (int i = 0; i < maxIteration; i++) {
        procruste(F_rest, V_rest, V_target, V_rest); //update V_rest mesh
        polyscope::registerSurfaceMesh("rest mesh", V_rest, F_rest);
    }
}

void callbackProcruste() {
    static int maxIteration = 50;
    ImGui::InputInt("max iter", &maxIteration);
    static float iteration = 0;
    if (ImGui::SliderFloat("run procruste", &iteration, 0, maxIteration)){
        procrusteIteration(iteration);
    }
}


int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    //create2DGridVariationsAndSave();

    //Rotate a mesh
    //Eigen::MatrixXd n_x_axis(n, 3), n_z_axis(n, 3);
    //for (int i = 0; i < n; i++) { // row
    //    n_x_axis.row(i) = Eigen::Vector3d(1, 0, 0);
    //}
    // 
    //for (int i = 0; i < n; i++) { // row
    //    n_z_axis.row(i) = Eigen::Vector3d(0, 0, 1);
    //}
    //Eigen::VectorXd angle = Eigen::VectorXd::Constant(n, 1, 90);
    //Eigen::MatrixXd V4 = igl::rotate_vectors(V1, angle, n_x_axis, n_z_axis);

    polyscope::init();
    polyscope::state::userCallback = callbackProcruste;
    polyscope::show();

    return 0;
}


//int main2(int argc, char **argv) {
//  //         //
//  // 2D grid //
//  //         //
//  Eigen::MatrixXd grid_V;
//  Eigen::MatrixXi grid_F;
//  create2DGridManual(10, 20, grid_V, grid_F);
//
//  // Options
//  polyscope::options::autocenterStructures = true;
//  polyscope::view::windowWidth = 1024;
//  polyscope::view::windowHeight = 1024;
//
//  // Initialize polyscope
//  polyscope::init();
//
//  // Register the mesh with Polyscope
//  polyscope::registerSurfaceMesh("input mesh", grid_V, grid_F);
//
//  Eigen::MatrixXd func(10, 20);
//  function(10, 20, func);
//  auto mesh = polyscope::getSurfaceMesh("input mesh");
//
//  func.transposeInPlace();
//  Eigen::VectorXd func_F(Eigen::Map<Eigen::VectorXd>(func.data(), func.cols() * func.rows()));
//  auto temp = mesh->addFaceScalarQuantity("test", func_F);
//
//  polyscope::show();
//
//  //         //
//  // 3D mesh //
//  //         //
//  
//  //std::string filename = "../spot.obj";
//  //std::cout << "loading: " << filename << std::endl;
//
//  //Eigen::MatrixXd origV;
//  //Eigen::MatrixXi origF;
//
//  // Read the mesh
//  //igl::readOBJ(filename, origV, origF);
//  ////Eigen::SparseMatrix<double> S;
//  //igl::loop(origV.rows(), origF, S, meshF);
//  //meshV = S * origV;
//
//  // Register the mesh with Polyscope
//  //polyscope::registerSurfaceMesh("input mesh", meshV, meshF);
//
//  //n = meshV.rows();// number of vertex in the mesh
//  //randN = Eigen::VectorXd::Random(n, 1);
//
//  // Add the callback
//  // polyscope::state::userCallback = callback;
//
//  // Show the gui
//  //polyscope::show();
//
//
//  return 0;
//}
