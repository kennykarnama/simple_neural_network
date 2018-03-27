#include "cu.h"

CommonUtility::CommonUtility(){

}

CommonUtility::~CommonUtility(){

}

void CommonUtility::printVector(vector< vector<string> > s){
	
	cout << " == PRINT VECTOR == " <<endl;

	for (int i = 0; i < s.size(); ++i)
	{
		/* code */

		vector<string> v_str = s.at(i);


		cout << " == START == [" << i << "]" << " = "<< endl;

		for (int j = 0; j < v_str.size(); ++j)
		{
			/* code */
			cout << v_str.at(j) << endl;
		}

		cout << " == END ==" << endl;

	}

	cout << " == END PRINT VECTOR == " << endl;
}

 void CommonUtility::printVector( vector < vector<double> > i){

		cout << " == PRINT VECTOR == " <<endl;

	for (int y = 0; y < i.size(); ++y)
	{
		/* code */

		vector<double> v_i = i.at(y);


		cout << " == START == [" << y << "]" << " = "<< endl;

		for (int j = 0; j < v_i.size(); ++j)
		{
			/* code */
			cout << v_i.at(j) << endl;
		}

		cout << " == END ==" << endl;

	}

	cout << " == END PRINT VECTOR == " << endl;
}

void CommonUtility::printVector( vector<arma::mat> m){

		cout << " == PRINT VECTOR == " <<endl;

	for (int y = 0; y < m.size(); ++y)
	{
		/* code */

		arma::mat sub_m = m.at(y);


		cout << " == START == [" << y << "]" << " = "<< endl;

		cout << sub_m << endl;

		cout << " == END ==" << endl;

	}

	cout << " == END PRINT VECTOR == " << endl;
}

void CommonUtility::printVector( vector<int> t){

		cout << " == PRINT VECTOR == " <<endl;

	for (int y = 0; y < t.size(); ++y)
	{
		/* code */

	


		cout << " == START == [" << y << "]" << " = "<< endl;

		cout << t.at(y) << endl;

		cout << " == END ==" << endl;

	}

	cout << " == END PRINT VECTOR == " << endl;
}

void CommonUtility::printVector( vector<double> t){

		cout << " == PRINT VECTOR == " <<endl;

	for (int y = 0; y < t.size(); ++y)
	{
		/* code */

	


		cout << " == START == [" << y << "]" << " = "<< endl;

		cout << t.at(y) << endl;

		cout << " == END ==" << endl;

	}

	cout << " == END PRINT VECTOR == " << endl;
}