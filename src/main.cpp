#include "polyscope/polyscope.h"

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


#include<Eigen/SparseCholesky>	

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


void reactionDiffusionExplicit(float t, float _alpha, float _beta, float _s, float _da, float _db) {
    using namespace Eigen;
    using namespace std;

    SparseMatrix<double> L, M;
    igl::cotmatrix(meshV, meshF, L); // returns V by V
    //igl::massmatrix(meshV, meshF, igl::MASSMATRIX_TYPE_VORONOI, M); // returns V by V
    //SimplicialLDLT<SparseMatrix<double> > solver;
    //solver.compute(M);//diagonal
    //L = solver.solve(L); // L = M^(-1)L

    VectorXd a = VectorXd::Constant(n, 1, 4) + randN;
    VectorXd b = VectorXd::Constant(n, 1, 4) + randN;

    //Converge to Turing's pattern
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
    auto temp = mesh->addVertexScalarQuantity("RD", a);
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


void addCurvatureScalar() {
  using namespace Eigen;
  using namespace std;

  VectorXd K;
  igl::gaussian_curvature(meshV, meshF, K);
  SparseMatrix<double> M, Minv;
  igl::massmatrix(meshV, meshF, igl::MASSMATRIX_TYPE_DEFAULT, M);
  igl::invert_diag(M, Minv);
  K = (Minv * K).eval();

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexScalarQuantity("gaussian curvature", K,
                                polyscope::DataType::SYMMETRIC);
}

void computeDistanceFrom() {
  Eigen::VectorXi VS, FS, VT, FT;
  // The selected vertex is the source
  VS.resize(1);
  VS << iVertexSource;
  // All vertices are the targets
  VT.setLinSpaced(meshV.rows(), 0, meshV.rows() - 1);
  Eigen::VectorXd d;
  igl::exact_geodesic(meshV, meshF, VS, FS, VT, FT, d);

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexDistanceQuantity(
          "distance from vertex " + std::to_string(iVertexSource), d);
}

void computeParameterization() {
  using namespace Eigen;
  using namespace std;

  // Fix two points on the boundary
  VectorXi bnd, b(2, 1);
  igl::boundary_loop(meshF, bnd);

  if (bnd.size() == 0) {
    polyscope::warning("mesh has no boundary, cannot parameterize");
    return;
  }

  b(0) = bnd(0);
  b(1) = bnd(round(bnd.size() / 2));
  MatrixXd bc(2, 2);
  bc << 0, 0, 1, 0;

  // LSCM parametrization
  Eigen::MatrixXd V_uv;
  igl::lscm(meshV, meshF, b, bc, V_uv);

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexParameterizationQuantity("LSCM parameterization", V_uv);
}

void computeNormals() {
  Eigen::MatrixXd N_vertices;
  igl::per_vertex_normals(meshV, meshF, N_vertices);

  polyscope::getSurfaceMesh("input mesh")
      ->addVertexVectorQuantity("libIGL vertex normals", N_vertices);
}

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;

  ImGui::PushItemWidth(100);

  // Curvature
  if (ImGui::Button("add curvature")) {
    addCurvatureScalar();
  }
  
  // Normals 
  if (ImGui::Button("add normals")) {
    computeNormals();
  }

  // Param
  if (ImGui::Button("add parameterization")) {
    computeParameterization();
  }

  // Geodesics
  if (ImGui::Button("compute distance")) {
    computeDistanceFrom();
  }

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
      reactionDiffusionExplicit(tRD, alpha, beta, s, da, db);
  }


  ImGui::SameLine();
  ImGui::InputInt("source vertex", &iVertexSource);

  ImGui::PopItemWidth();
}

int main(int argc, char **argv) {

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  std::string filename = "../spot.obj";
  std::cout << "loading: " << filename << std::endl;

  // Read the mesh
  igl::readOBJ(filename, meshV, meshF);

  // Register the mesh with Polyscope
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  n = meshV.rows();// number of vertex in the mesh
  randN = Eigen::VectorXd::Random(n, 1);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
