#include "nnbp.h"
#include "acfunc.h"

NeuralNetwork::NeuralNetwork(){
	squaredSum = 0.0;
}

NeuralNetwork::~NeuralNetwork(){

}

void NeuralNetwork::init(int inputNodes, int hiddenNodes, int outputNodes, double learningRate){
	
	inNodes = inputNodes;

	hNodes = hiddenNodes;

	oNodes = outputNodes;

	lr = learningRate;

	wih = arma::randu<arma::mat>(hNodes,inNodes);

	initWeight(0.0,sqrt(hNodes),wih);

	who = arma::randu<arma::mat>(oNodes,hNodes);

	initWeight(0.0,sqrt(oNodes),who);


}

void NeuralNetwork::train(arma::mat inputs_list, arma::mat targets_list){

	arma::mat inputs_listT = inputs_list.t();

	arma::mat targets_listT = targets_list.t();

	// same as query function

	arma::mat hidden_input = wih * inputs_listT;

	arma::mat hidden_output = doActivationFunction(hidden_input);

	arma::mat final_input = who * hidden_output;

	arma::mat final_output = doActivationFunction(final_input);

	// calculate output error

	arma::mat output_errors = targets_listT - final_output;

	// double squared_sum = 0.0;

	// double length = output_errors.n_rows * output_errors.n_cols;

	for(int y = 0; y < output_errors.n_rows; y++){
		
		for(int x = 0; x < output_errors.n_cols; x++){
			
			squaredSum+= pow(output_errors.at(y,x),2);

		}
	}

	

	//std::cout << output_errors << std::endl;

	// calculate errorsHidden = whoT * output_errors;

	arma::mat whoT  = who.t();

	arma::mat hidden_errors = whoT * output_errors;

	//update who

	arma::mat hidden_outputT = hidden_output.t();

	arma::mat firstPart = (output_errors % final_output % (1.0 - final_output));

	// std::cout << "hot "<< hidden_outputT << std::endl;

	// std::cout << "test "<<test << std::endl;

	arma::mat resultFirstPart = firstPart * hidden_outputT;

	// ((output_errors % final_output % (1.0 - final_output)) * hidden_outputT)

	 who += lr * resultFirstPart;

	 //std::cout << "After updated who "<< who << std::endl;

	// // update wih

	 arma::mat secondPart = (hidden_errors % hidden_output % (1.0 - hidden_output));

	 //std::cout << "secondPart "<< secondPart << std::endl;

	 arma::mat resultSecondPart =  secondPart * inputs_list;

	 // ((hidden_errors % hidden_output % (1.0 - hidden_output)) * inputs_listT)
	 
	 wih += lr * resultSecondPart;

 	//std::cout << "After updated wih "<< wih << std::endl;

	 // std::cout << "== MEENG ==" << std::endl;

	 // arma::uword mati = final_output.index_max();

	 // std::cout << mati << std::endl;
	 // std::cout << final_output << std::endl;

	 // std::cout << " == END MEENG == " << std::endl; 



}

arma::mat NeuralNetwork::query(arma::mat input){

	arma::mat inputT = input.t();

	// calculate input hidden

	arma::mat hidden_input = wih * inputT;

	arma::mat hidden_output = doActivationFunction(hidden_input);

	//std::cout << "== HIDDEN OUTPUT =="<< std::endl;

	// std::cout << wih << std::endl;

	// std::cout << hidden_output << std::endl;

	// std::cout << "== END HIDDEN OUTPUT=="<<std::endl;




	// calculate hidden output

	arma::mat final_input = who * hidden_output;

	arma::mat final_output = doActivationFunction(final_input);

	// 	std::cout << "==  OUTPUT TARGET =="<< std::endl;

	// std::cout << who << std::endl;

	// std::cout << final_output << std::endl;

	// std::cout << "== END OUTPUT TARGET=="<<std::endl;



	return final_output;

}

int NeuralNetwork::getScoreCard(arma::mat input, int targetLabel){

	std::vector<int> scoreCard;

	arma::mat inputT = input.t();

	// calculate input hidden

	arma::mat hidden_input = wih * inputT;

	arma::mat hidden_output = doActivationFunction(hidden_input);


	// calculate hidden output

	arma::mat final_input = who * hidden_output;

	arma::mat final_output = doActivationFunction(final_input);

	arma::uword network_answer = final_output.index_max();

	std::cout <<"LABEL " << targetLabel <<" " << " answer " << network_answer << std::endl;

	if((int)network_answer == targetLabel){
		return 1;
	}
	else{
		return 0;
	}

	//return scoreCard;
	
}

void NeuralNetwork::printWeights(){

	std::cout << " == Weight Input Hidden == " << std::endl;
	std::cout << wih << std::endl;
	std::cout << " == END Weight Input Hidden == "<<std::endl;

	std::cout << " == Weight Hidden Output == " << std::endl;
	std::cout << who << std::endl;
	std::cout << " == END Weight Hidden Output == "<<std::endl;


}

void NeuralNetwork::initWeight(double centerOfDistribution, double standardDeviation, arma::mat &weightMat){

	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	
  	std::default_random_engine generator (seed);

  	std::normal_distribution<double> distribution (centerOfDistribution,standardDeviation);
   

	for (int i = 0; i < weightMat.n_rows; ++i)
	{
		/* code */
		for (int j = 0; j < weightMat.n_cols; ++j)
		{
			/* code */
			weightMat.at(i,j) =  distribution(generator);

		}
	}




}

arma::mat NeuralNetwork::doActivationFunction(const arma::mat &m){

	arma::mat O = arma::mat(m.n_rows,m.n_cols);

	// do sigmoid function

	for (int i = 0; i < m.n_rows; ++i)
	{
		/* code */
		for (int j = 0; j < m.n_cols; ++j)
		{
			/* code */
			O.at(i,j) = ActivationFunction::sigmoid(m.at(i,j));
		}
	}

	return O;

}

arma::mat NeuralNetwork::getWih(){

	return wih;

}

arma::mat NeuralNetwork::getWho(){

	return who;

}

double NeuralNetwork::getSquaredSum(){
	return squaredSum;
}

void NeuralNetwork::setSquaredSum(double val){
	squaredSum = val;
}