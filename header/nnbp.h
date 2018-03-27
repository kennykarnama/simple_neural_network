#include <iostream>
#include <armadillo>
#include <chrono>
#include <random>
#include <cmath>

#ifndef NNBP_H
#define NNBP_H
	class NeuralNetwork
	{
	public:
		NeuralNetwork();
		~NeuralNetwork();

		void init(int,int,int,double);

		void train(arma::mat, arma::mat);

		arma::mat query(arma::mat);

		int getScoreCard(arma::mat, int);

		double getSquaredSum();

		void printWeights();

		void setSquaredSum(double);

		arma::mat getWih();

		arma::mat getWho();


	
		/* data */

	private:
		int inNodes;
		int hNodes;
		int oNodes;
		double lr;
		arma::mat wih;
		arma::mat who;
		
		void initWeight(double,double, arma::mat &);

		arma::mat doActivationFunction(const arma::mat &);

		double squaredSum;


		

	};
#endif