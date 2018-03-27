#include "mytraining.h"

MyTraining::MyTraining(){

}

MyTraining::~MyTraining(){

}

void MyTraining::build_str_data_training( std::string alamatFile ){


	// membaca file csv, diubah menjadi vector<string>

	io::LineReader in(alamatFile);

	std::vector<std::string> lineCSV;

	while(char* line = in.next_line() ){

		//std::cout << "tet";
		std::string newString (line);

		// std::cout << " == START LINE ==" << std::endl;
		// std::cout << newString << std::endl;
		
		lineCSV.push_back(newString);

		// std::cout << " == END LINE == "<<std::endl;

	}

	// buat menjadi vector< vector<string> >


	for (int i = 0; i < lineCSV.size(); ++i)
	{
		/* code */
		std::string text = lineCSV.at(i);

		std::vector<std::string> results;

		

		boost::split(results, text, [](char c){return c == ',';});

		str_data_training.push_back(results);

		//std::cout << "Besar string " << results.size() << std::endl;
	}

	



}

std::vector < std::vector<std::string> > MyTraining::getStrDataTraining(){

	return str_data_training;

}

std::vector < std::vector< double> > MyTraining::getDataTraining(){
	return data_training;
}

std::vector <arma::mat> MyTraining::getInput(){
	return input;
}

std::vector<int> MyTraining::getTargetLabel(){
	return targetLabel;
}

std::vector < arma::mat > MyTraining::getTarget(){
	return target;
}


void MyTraining::build_data_training(){

	for (int i = 0; i < str_data_training.size(); ++i)
	{
		/* code */
		std::vector < std::string>  sub_str_data_training = str_data_training.at(i);

		// iterasi sepanjang sub_str_data_training
		// ubah setiap elemen menjadi int 
		// menggunakan stoi

		std::vector < double > els;

		int target_label = std::stoi(sub_str_data_training.at(0),nullptr,10);

		targetLabel.push_back(target_label);

		for (int j = 1; j < sub_str_data_training.size(); ++j)
		{
			/* code */

			std::string::size_type sz;

			double el = std::stod(sub_str_data_training.at(j), &sz);

			el = (el / 255.0) * 0.99 + 0.01;

			els.push_back(el);
		}

		// insert ke dalam vector data_training

		data_training.push_back(els);
	}

}

void MyTraining::build_input(){
	for (int i = 0; i < data_training.size(); ++i)
	{
		/* code */
		std::vector<double> v_data_training = data_training.at(i);

		// init arma mat

		arma::mat input_list = arma::randu<arma::mat>(1,v_data_training.size());


		// iterate over input_list 

		int puter = 0;

		for (int row = 0; row < input_list.n_rows; ++row)
		{
			/* code */
			for (int col = 0; col < input_list.n_cols; ++col)
			{
				/* code */
				input_list.at(row,col) = v_data_training.at(puter);

				puter++;
			}
		}

		input.push_back(input_list);

	}
}

void MyTraining::build_output(int numOfOutputLayers){

	// foreach test case

	for (int i = 0; i < input.size(); ++i)
	{
		/* code */
		arma::mat target_list(1,numOfOutputLayers);

		target_list.fill(0.01);

		target_list.at(0,targetLabel.at(i)) = 0.99;

		target.push_back(target_list);

	}

	

}