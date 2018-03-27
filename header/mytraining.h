#include <armadillo>
#include <iostream>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include "csv.h"

#ifndef MYTRAINING_H
#define MYTRAINING_H
	class MyTraining
	{
	public:
		
		MyTraining();
		
		~MyTraining();

		void build_str_data_training( std::string );

		void build_data_training();

		void build_input();

		void build_target();

		void build_output(int);

		std::vector < std::vector<std::string> > getStrDataTraining();

		std::vector < std::vector< double> > getDataTraining();

		std::vector < arma::mat > getInput();

		std::vector< int > getTargetLabel();

		std::vector < arma::mat > getTarget();

		

	private:
	
	std::vector< std::vector<std::string> > str_data_training;

	std::vector< std::vector<double> > data_training;

	std::vector<arma::mat> input;

	std::vector<arma::mat> target;

	std::vector<int> targetLabel;


	
		/* data */
	};
#endif

