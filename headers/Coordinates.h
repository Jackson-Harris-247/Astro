#define EQUATORIAL 0
#define GALACTIC 1
#define GALCTOCENTRIC 2

#include "Conversions.h"

using namespace Eigen;
using namespace std;

//PSVector Formating:
//Equatorial (RA, DEC, PLAX, PMRA, PMDEC, RV)
//Right Ascension, Declination, Parallax, Proper Motion Right Ascension, Proper Motion Declination, Radial Velocity
//Galactic (l, b, d, Vlos, PMl*, PMb)
//Longitude, Latitude, Distance from Sun, Velocity in Line of Sight, Proper Motion Longitude, Proper Motion Latitude
//Galactocentric (X, Y, Z, Vx, Vy, Vz)
//X, Y, Z, Velocity X, Velocity Y, Velocity Z

class Coordinate {
	public:
	int MODE = EQUATORIAL;
	VectorXd PSVector;
	Vector2d propogatedError;

	double pmra_error, pmdec_error, pmra_pmdec_corr;

	double cosPsi, sinPsi;

	Coordinate() = default;

	Coordinate(int nMODE, VectorXd nPSVector) {
		MODE = nMODE;
		PSVector = nPSVector;
	}

	Coordinate(double ra, double dec, double plax, double pmra, double pmdec, double rv, double Npmra_error, double Npmdec_error, double Npmra_pmdec_corr) : PSVector(6) {
		PSVector << ra,dec,plax,pmra,pmdec,rv;

		pmra_error = Npmra_error;
		pmdec_error = Npmdec_error;
		pmra_pmdec_corr = Npmra_pmdec_corr;

		MODE = EQUATORIAL;
		//cout << PSVector;
	}

	VectorXd getPSVector(void) {
		return PSVector;
	}

	Vector2d calcError() {
		double RA = PSVector[0];
		double DEC = PSVector[1];
		double b = asinDEG(sinDEG(DEC)*sinDEG(DNGP)+cosDEG(DEC)*cosDEG(DNGP)*cosDEG(RA-ANGP));
		return propogateErrors( calcCosPsi(RA, DEC, b), calcSinPsi(RA, b), pmra_error,pmdec_error,pmra_pmdec_corr);
	}
};
