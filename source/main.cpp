#include "nnbp.h"
#include "acfunc.h"
#include "csv.h"
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include "cu.h"
#include "mytraining.h"
#include "algorithm"
#include "chrono"
#include "random"
#include <ctime>


int main(int argc, char const *argv[])
{

	

	std::time_t now = std::time(0);

	char* dt = std::ctime(&now);

	std::string sekarang(dt);

	 std::replace(sekarang.begin(),sekarang.end(),' ','_');

	std::cout << "TRAINING START AT " << sekarang << std::endl;


	/* code */

	// inisialisasi nn nya

	// parameter pertama itu jumlah input node

	// parameter kedua itu jumlah hidden node

	// parameter ketiga itu jumlah output node

	// parameter keempat, learning rate

	 NeuralNetwork nn;

	nn.init(784,100,26,0.01);

	double before =  0.0;

	double current = 0.0;

	// iterasi sepanjang epoch

	int maxEpoch = 5000;

	for (int e = 0; e < maxEpoch; ++e)
	{
		/* code */

		std::cout << "Training ke " << (e+1) << std::endl;

		// buka data training berdasarkan file csv

		io::LineReader in(argv[1]);



	std::vector<std::string> lineCSV;

	int jumlahSampel = 0;

	before = current;

	while(char* line = in.next_line() ){

		//std::cout << "tet";
		std::string newString (line);

		// int actualValue = newString.at(0) - '0';

		std::vector<std::string> results;

		boost::split(results, line, [](char c){return c == ',';});

		std::vector<double> els;

		int actualValue = std::stoi(results.at(0));

		for (int j = 1; j < results.size(); ++j)
		{
			/* code */

			std::string::size_type sz;

			double el = std::stod(results.at(j), &sz);

			// di resampling

			el = (el / 255.0) * 0.99 + 0.01;

			els.push_back(el);
		}

		arma::mat input_list = arma::randu<arma::mat>(1,784);


		// iterate over input_list 

		int puter = 0;

		for (int row = 0; row < input_list.n_rows; ++row)
		{
			/* code */
			for (int col = 0; col < input_list.n_cols; ++col)
			{
				/* code */
				input_list.at(row,col) = els.at(puter);

				puter++;
			}
		}

		//std::cout <<"input list "<< input_list << std::endl;

		arma::mat target_list(1,26);

		target_list.fill(0.01);

		target_list.at(0,actualValue-1) = 0.99;

		//std::cout << "Target list untuk " << actualValue << target_list << std::endl;

		// di training

	
		 nn.train(input_list,target_list);



		jumlahSampel++;

	}

	 double squaredSum = nn.getSquaredSum();

	 std::cout << "Jumlah jumlahSampel "<< jumlahSampel << std::endl;

	 double mse = (squaredSum) / (jumlahSampel*1.0);

	 current = mse;

	 double selisih = current - before;

	 std::cout << "MSE  " << mse << std::endl;

	 std::cout <<"Before " << before << std::endl;
	 
	 std::cout << "Selisih MSE "<< selisih << std::endl;

	 if(mse < 0.02){


	 	break;
	 }

	 nn.setSquaredSum(0.0);

	 	// uji kasusnya

	io::LineReader in_testing(argv[2]);

	std::vector<int> scoreCard;

	int barisKe = 1;

	while(char* line_testing = in_testing.next_line()){

		std::string newString_testing (line_testing);

		//int actualValue_testing = newString_testing.at(0) - '0';

		// int actualValue_testing = std::stoi(newString_testing.at())
		// std::cout << "Baris ke" << barisKe << " "<< actualValue_testing << std::endl;

		barisKe++;

		std::vector<std::string> results_testing;

		boost::split(results_testing, line_testing, [](char c){return c == ',';});

		std::vector<double> els_testing;

		int actualValue_testing = std::stoi(results_testing.at(0));

		//std::cout << "Baris ke " << barisKe << " " << actualValue_testing << std::endl;

		for (int j = 1; j < results_testing.size(); ++j)
		{
			/* code */

			std::string::size_type sz_testing;

			double el_testing = std::stod(results_testing.at(j), &sz_testing);

			// di resampling

			el_testing = (el_testing / 255.0) * 0.99 + 0.01;

			els_testing.push_back(el_testing);
		}

		arma::mat input_list_testing = arma::randu<arma::mat>(1,784);


		// iterate over input_list 

		int puter_testing = 0;

		for (int row_testing = 0; row_testing < input_list_testing.n_rows; ++row_testing)
		{
			/* code */
			for (int col_testing = 0; col_testing < input_list_testing.n_cols; ++col_testing)
			{
				/* code */
				input_list_testing.at(row_testing,col_testing) = els_testing.at(puter_testing);

				puter_testing++;
			}
		}

		//std::cout <<"input list "<< input_list << std::endl;

	int status_jawaban = nn.getScoreCard(input_list_testing,actualValue_testing-1);	

	scoreCard.push_back(status_jawaban);

	}


	
	
	double jumlahBenar = 0.0;

	double jumlahKasusUji = scoreCard.size();

	for (int i = 0; i < scoreCard.size(); ++i)
	{
		/* code */
		if(scoreCard.at(i) == 1){
			jumlahBenar++;
		}
	}

	std::cout << "jumlah benar " << jumlahBenar << std::endl;

	std::cout << "jumlah kasus " << jumlahKasusUji << std::endl;

	double akurasi = ((jumlahBenar * 1.0) / (jumlahKasusUji * 1.0)) * 100;

	std::cout << "Akurasinya adalah " << akurasi << " % " << std::endl;
	


	}

	// get trained wih and who

	arma::mat trained_wih = nn.getWih();

	arma::mat trained_who = nn.getWho();

	// save to arma bin

	std::string wihFileName = sekarang+"_wih.bin";

	std::string whoFileName = sekarang+"_who.bin";

	trained_wih.save(wihFileName);

	trained_who.save(whoFileName);

		 	// uji kasusnya

	io::LineReader in_testing(argv[3]);

	std::vector<int> scoreCard;

	int barisKe = 1;

	while(char* line_testing = in_testing.next_line()){

		std::string newString_testing (line_testing);

		//int actualValue_testing = newString_testing.at(0) - '0';

		// int actualValue_testing = std::stoi(newString_testing.at())
		// std::cout << "Baris ke" << barisKe << " "<< actualValue_testing << std::endl;

		barisKe++;

		std::vector<std::string> results_testing;

		boost::split(results_testing, line_testing, [](char c){return c == ',';});

		std::vector<double> els_testing;

		int actualValue_testing = std::stoi(results_testing.at(0));

		//std::cout << "Baris ke " << barisKe << " " << actualValue_testing << std::endl;

		for (int j = 1; j < results_testing.size(); ++j)
		{
			/* code */

			std::string::size_type sz_testing;

			double el_testing = std::stod(results_testing.at(j), &sz_testing);

			// di resampling

			el_testing = (el_testing / 255.0) * 0.99 + 0.01;

			els_testing.push_back(el_testing);
		}

		arma::mat input_list_testing = arma::randu<arma::mat>(1,784);


		// iterate over input_list 

		int puter_testing = 0;

		for (int row_testing = 0; row_testing < input_list_testing.n_rows; ++row_testing)
		{
			/* code */
			for (int col_testing = 0; col_testing < input_list_testing.n_cols; ++col_testing)
			{
				/* code */
				input_list_testing.at(row_testing,col_testing) = els_testing.at(puter_testing);

				puter_testing++;
			}
		}

		//std::cout <<"input list "<< input_list << std::endl;

	int status_jawaban = nn.getScoreCard(input_list_testing,actualValue_testing);	

	scoreCard.push_back(status_jawaban);

	}


	
	
	double jumlahBenar = 0.0;

	double jumlahKasusUji = scoreCard.size();

	for (int i = 0; i < scoreCard.size(); ++i)
	{
		/* code */
		if(scoreCard.at(i) == 1){
			jumlahBenar++;
		}
	}

	std::cout << "jumlah benar " << jumlahBenar << std::endl;

	std::cout << "jumlah kasus " << jumlahKasusUji << std::endl;

	double akurasi = ((jumlahBenar * 1.0) / (jumlahKasusUji * 1.0)) * 100;

	std::cout << "Akurasinya adalah " << akurasi << " % " << std::endl;


	
	

	return 0;
}

