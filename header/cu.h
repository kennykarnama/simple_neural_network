#include <iostream>
#include <vector>
#include <string>
#include <armadillo>

using namespace std;

#ifndef CU_H
#define CU_H

	class CommonUtility
	{
	public:
		CommonUtility();
		
		~CommonUtility();

		static void printVector( vector<vector<string> >);

		static void printVector( vector<vector<double> >);
		
		static void printVector ( vector <arma::mat> );

		static void printVector ( vector< int> );

		static void printVector ( vector< double > );
		/* data */
	};
#endif
