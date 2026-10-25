#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "headers/CelestialObject.h"
#include "headers/HDBSCAN.h"

using namespace std;
namespace py = pybind11;

vector<CelestialObject> mainProcessing(vector<CelestialObject>& dataVector) {
	const int64_t n = dataVector.size();

	//Convert the Equatorial Coordinates to Galactocentric
	#pragma omp parallel for schedule(static)
	for(int64_t i = 0; i < n; i++) {
		dataVector[i].GACoords = Coordinate(2, GAfromEQ(dataVector[i].EQCoords.PSVector));
		dataVector[i].GCCoords = Coordinate(2, GCfromGA(dataVector[i].GACoords.PSVector));
	}
	cout << "Calculated Galactic Coordinates" << '\n';

	// Convert CelestialObjects to MatrixXd for clustering
	MatrixXd dataMatrix(dataVector.size(), 6); // 6 coordinates: RA, DEC, parallax, pmra, pmdec, radial_velocity

	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < n; i++) {
		const CelestialObject& obj = dataVector[i];
		dataMatrix(i, 0) = obj.GCCoords.PSVector(0); // X
		dataMatrix(i, 1) = obj.GCCoords.PSVector(1); // Y
		dataMatrix(i, 2) = obj.GCCoords.PSVector(2); // Z
		dataMatrix(i, 3) = obj.GCCoords.PSVector(3); // Vx
		dataMatrix(i, 4) = obj.GCCoords.PSVector(4); // Vy
		dataMatrix(i, 5) = obj.GCCoords.PSVector(5); // Vz
	}
	cout << "Created Data Matrix" << '\n';
	cout << "Starting HDBSCAN" << '\n';

	// Apply HDBSCAN clustering
	HDBSCAN hdbscan;
	vector<int> labels = hdbscan.clusterAdvanced(dataMatrix, 2, 4);
	cout << "Finished HDBSCAN" << '\n';

	int num_clusters = 0;
	for(int label : labels) {
    	if (label >= num_clusters) {
        	num_clusters = label + 1;
    	}
	}

	vector<CelestialObject> filteredVector;
	filteredVector.reserve(dataVector.size());

	for (size_t i = 0; i < dataVector.size(); i++) {
    	//Just keeps the noise
    	if(labels[i] == -1) {
        	filteredVector.push_back(dataVector[i]);
    	}
	}

	for (size_t i = 0; i < filteredVector.size(); i++) {
		double X = filteredVector[i].GCCoords.PSVector[0];
		double Vx = filteredVector[i].GCCoords.PSVector[3];
		double Y = filteredVector[i].GCCoords.PSVector[1];
		double Vy = filteredVector[i].GCCoords.PSVector[4];
		double R = sqrt(X*X+Y*Y);
		filteredVector[i].galacticRadius = R;
		filteredVector[i].vPhi = (X*Vy-Y*Vx)/R;
	}

	cout << "HDBSCAN found " << num_clusters << " clusters" << '\n';
	cout << "Filtered down to " << filteredVector.size() << " Celestial Objects" << '\n';
	return filteredVector;
}

PYBIND11_MODULE(Transformations, handle) {
	handle.doc() = "Python module written in C++ for astrophysics transformations"	  ;

	py::class_<Coordinate>(handle, "Coordinate")
	.def(py::init<double,double,double,double,double,double,double,double,double>())
	.def("getPSVector", &Coordinate::getPSVector)
	.def("calcError", &Coordinate::calcError)
	;

	py::class_<CelestialObject>(handle, "CelestialObject")
	.def(py::init<Coordinate>())
	.def(py::init<string, Coordinate>())
	.def("getGalacticRadius", &CelestialObject::getGalacticRadius)
	.def("getVPhi", &CelestialObject::getVPhi)
	.def("getEQCoordinate", &CelestialObject::getEQCoordinate)
	.def("getGACoordinate", &CelestialObject::getGACoordinate)
	.def("getGCCoordinate", &CelestialObject::getGCCoordinate)
	;

	handle.def("searchDist", &searchDist);
	handle.def("mainProcessing", &mainProcessing, py::call_guard<py::gil_scoped_release>());
}
