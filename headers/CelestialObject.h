#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <iostream>
#include <numbers>

#include "Coordinates.h"

using namespace std;

//The object that will hold the data for each datapoint
class CelestialObject {

    public:
    Coordinate GACoords, EQCoords, GCCoords;
	//Galactic, Equatorial, Galactocentric

	double galacticRadius;
	double vPhi;

	string name;

    Eigen::MatrixXd covarianceGA;
    Eigen::MatrixXd covarianceGC;

	double getGalacticRadius(void) {
		return galacticRadius;
	}

	double getVPhi(void) {
		return vPhi;
	}

	Coordinate getEQCoordinate(void) {
		return EQCoords;
	}

	Coordinate getGACoordinate(void) {
		return GACoords;
	}

	Coordinate getGCCoordinate(void) {
		return GCCoords;
	}

    //Create object without a name
    CelestialObject(Coordinate ncoords) {
        EQCoords = ncoords;
    }

    //Create object without a name
    CelestialObject(string nname, Coordinate ncoords) {
        name = nname;
        EQCoords = ncoords;
    }
};
