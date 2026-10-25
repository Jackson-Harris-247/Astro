#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#define DNGP 27.12825
#define ANGP 192.85948
#define LNCP 122.93192

#define R0 8.122
#define Z0 0.0208
#define KCONV 4.74057

#define DEGTORAD (3.141592653589/180.0)
#define RADTODEG (1.0/DEGTORAD)

using namespace Eigen;

double asinDEG(double x) {
    return RADTODEG * asin(x);
}

double sinDEG(double x) {
    return sin(DEGTORAD * x);
}

double cosDEG(double x) {
    return cos(DEGTORAD * x);
}

double atan2DEG(double x, double y) {
    return RADTODEG * atan2(x,y);
}

//Gets the distance to search for the degree
double searchDist(double l) {
	double n = 2.0;
	double a = pow(14.0,n);
	double b = pow(2.0,n);
	double d = 14.0-R0;
	double numerator = -b*d + sqrt(
				a*b*b*pow(cosDEG(l),2) +
				a*a*b*pow(sinDEG(l),2) -
				a*b*d*d*pow(sinDEG(l),2)
				);
	double denominator = b*pow(cosDEG(l),2) + a*pow(sinDEG(l),2);
	return (numerator/denominator);
}

MatrixXd Jacobian(double RA, double DEC) {
	MatrixXd a(2,2);
	return a;
}

double calcCosPsi(double RA, double DEC, double b) {
	return (sinDEG(DNGP)*cosDEG(DEC)-cosDEG(DNGP)*sinDEG(DEC)*cosDEG(RA-ANGP))/(cosDEG(b));
}

double calcSinPsi(double RA, double b) {
	return (cosDEG(DNGP)*sinDEG(RA-ANGP))/(cosDEG(b));
}

VectorXd GAfromEQ(const VectorXd& EQ) {
	double RA = EQ(0);
	double DEC = EQ(1);
	double PLAX = EQ(2);
	double PMRA = EQ(3);
	double PMDEC = EQ(4);
	double RV = EQ(5);

    // Intermediaries
	double y = cosDEG(DEC) * sinDEG(RA - ANGP);
	double x = sinDEG(DEC) * cosDEG(DNGP) - cosDEG(DEC) * sinDEG(DNGP) * cosDEG(RA - ANGP);

	//Positions
	double l = fmod(LNCP - atan2DEG(y, x) + 360.0, 360.0);
	double b = asinDEG(sinDEG(DEC) * sinDEG(DNGP) + cosDEG(DEC) * cosDEG(DNGP) * cosDEG(RA - ANGP));
	double d = 1.0 / PLAX; // Distance in kpc

	//Intermediaries
	double cosPsi = calcCosPsi(RA, DEC, b);
	double sinPsi = calcSinPsi(RA, b);

	//Velocity terms
    double PMl = cosPsi * PMRA + sinPsi * PMDEC;
    double PMb = -sinPsi * PMRA + cosPsi * PMDEC;

	//Push to output
	VectorXd GA(6);
	GA << l, b, d, RV, PMl, PMb;
	return GA;
}

VectorXd GCfromGA(const VectorXd& GA) {
	double l = GA(0);
	double b = GA(1);
	double d = GA(2);
	double Vlos = GA(3);
	double PMl = GA(4);
	double PMb = GA(5);

	double V_l = KCONV * d * PMl;
	double V_b = KCONV * d * PMb;

	VectorXd VGA(3);
	VGA << V_l, V_b, Vlos;

	double xSun = d * cosDEG(b) * cosDEG(l);
	double ySun = d * cosDEG(b) * sinDEG(l);
	double zSun = d * sinDEG(b);

	VectorXd Vsun(3);
	Vsun << 12.9, 245.6, 7.78;

	MatrixXd T(3,3);
	T << -sinDEG(l), -cosDEG(l)*sinDEG(b), cosDEG(l)*cosDEG(b),
		  cosDEG(l), -sinDEG(l)*sinDEG(b), sinDEG(l)*cosDEG(b),
		  0.0,        cosDEG(b),           sinDEG(b);

	VectorXd VRelativeToSun = T * VGA;
	VectorXd VGalactic = VRelativeToSun + Vsun;

	//Positions
	double X = xSun - R0;
	double Y = ySun;
	double Z = zSun + Z0;

	//Push to output
	VectorXd GC(6);
	GC << X, Y, Z, VGalactic(0), VGalactic(1), VGalactic(2);
	return GC;
}

VectorXd GCfromEQ(const VectorXd& EQ) {
	return GCfromGA(GAfromEQ(EQ));
}

Vector2d propogateErrors(double cosPsi, double sinPsi, double pmra_error, double pmdec_error, double pmra_pmdec_corr) {
	double cov = pmra_pmdec_corr * pmra_error * pmdec_error;
	double varL = cosPsi*cosPsi * pmra_error*pmra_error + sinPsi*sinPsi * pmdec_error*pmdec_error + 2*cosPsi*sinPsi*cov;
	double varB = sinPsi*sinPsi * pmra_error*pmra_error + cosPsi*cosPsi * pmdec_error*pmdec_error - 2*cosPsi*sinPsi*cov;
	Vector2d errors;
	errors << sqrt(std::max(0.0, varL)), sqrt(std::max(0.0, varB));
    return errors;
}
