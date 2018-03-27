#include "acfunc.h"

ActivationFunction::ActivationFunction(){

}

ActivationFunction::~ActivationFunction(){

}

double ActivationFunction::sigmoid(double x){
	
	double hasil = (1.0) / ((1.0)+exp(x * -1.0));

	return hasil;
}