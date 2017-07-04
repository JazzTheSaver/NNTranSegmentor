#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
	//required
	int beam;
	int maxlength;
	int action_num;
	dtype delta;
	unordered_set<string> dicts;  // dictionary in order to extract iv/oov features.


	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization
	dtype dropProb;

	int char_dim; 
	int chartype_dim;
	int bichar_dim;
	int word_dim;
	int action_dim;

	bool char_tune;
	bool bichar_tune;
	bool word_tune;

	int char_context;
	int char_repsentation_dim;
	int char_window_dim;

	int char_hidden_dim;  
	int word_hidden_dim;
	int action_hidden_dim;

	int char_lstm_dim;
	int word_lstm_dim;
	int action_lstm_dim;

	int sep_hidden_dim;
	int app_hidden_dim;

public:
	HyperParams(){
		maxlength = max_sentence_clength + 1;
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		//please specify dictionary outside
		//please sepcify char_dim, word_dim and action_dim outside.
		beam = opt.beam;
		delta = opt.delta;
		bAssigned = true;

		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		dropProb = opt.dropProb;

		char_dim = opt.charEmbSize;
		bichar_dim = opt.bicharEmbSize;
		chartype_dim = opt.charTypeEmbSize;
		word_dim = opt.wordEmbSize;
		action_dim = opt.actionEmbSize;

		char_tune = opt.charEmbFineTune;
		bichar_tune = opt.bicharEmbFineTune;
		word_tune = opt.wordEmbFineTune;

		char_context = opt.charcontext;
		char_repsentation_dim = char_dim + bichar_dim + chartype_dim;
		char_window_dim = (2 * char_context + 1) * char_repsentation_dim;

		char_hidden_dim = opt.charHiddenSize;
		word_hidden_dim = opt.wordHiddenSize;
		action_hidden_dim = opt.actionHiddenSize;

		char_lstm_dim = opt.charRNNHiddenSize;
		word_lstm_dim = opt.wordRNNHiddenSize;
		action_lstm_dim = opt.actionRNNHiddenSize;

		sep_hidden_dim = opt.sepHiddenSize;
		app_hidden_dim = opt.appHiddenSize;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


public:

	void print(){
	std::cout << "======= Hyper Params =======" << std::endl;
	std::cout << "beam = "<< beam << std::endl;
	std::cout << "maxlength = "<< maxlength << std::endl;
	std::cout << "action_num = " << action_num << std::endl;
	std::cout << "delta = " << delta << std::endl;

	for(unordered_set<string>::iterator it = dicts.begin();it != dicts.end();it++){
		std::cout << "dicts = " << *it << std::endl;
	}
	std::cout << "nnRegular = " << nnRegular << std::endl;
	std::cout << "adaAlpha = "<< adaAlpha << std::endl;
	std::cout << "adaEps = "<< adaEps << std::endl;
	std::cout << "dropProb = " << dropProb << std::endl;

	std::cout << "char_dim = "<< char_dim << std::endl;
	std::cout << "chartype_dim = " << chartype_dim << std::endl;
	std::cout << "bichar_dim = " << bichar_dim << std::endl;
	std::cout << "word_dim = " << word_dim << std::endl;
	std::cout << "action_dim = " << action_dim << std::endl;

	std::cout << "char_tune = " << char_tune << std::endl;
	std::cout << "bichar_tune = " << bichar_tune << std::endl;
	std::cout << "word_tune = " << word_tune << std::endl;

	std::cout << "char_context = " << char_context << std::endl;
	std::cout << "char_represention_dim = "<< char_repsentation_dim << std::endl;
	std::cout << "char_window_dim = "<< char_window_dim << std::endl;

	std::cout << "char_hidden_dim = " << char_hidden_dim << std::endl;
	std::cout << "word_hidden_dim = " << word_hidden_dim << std::endl;
	std::cout << "action_hidden_dim = " << action_hidden_dim << std::endl;

	std::cout << "char_lstm_dim = " << char_lstm_dim << std::endl;
	std::cout << "word_lstm_dim = " << word_lstm_dim << std::endl;
	std::cout << "action_lstm_dim = " << action_lstm_dim << std::endl;


	std::cout << "app_hidden_dim = " << app_hidden_dim << std::endl;
	std::cout << "sep_hidden_dim = " << sep_hidden_dim << std::endl;
	std::cout << "=======================" << std::endl;

	}
	void saveModel(std::ofstream& os)const {
		os << beam << std::endl;
		os << maxlength << std::endl;
		os << action_num << std::endl;
		os << delta << std::endl;

		os << dicts.size() << std::endl;
		for (unordered_set<string>::iterator it = dicts.cbegin(); it != dicts.end(); it++) {
			os << *it<< std::endl;
		}

		os << nnRegular << std::endl;
		os << adaAlpha << std::endl;
		os << adaEps << std::endl;
		os << dropProb << std::endl;

		os << char_dim << std::endl;
		os << chartype_dim << std::endl;
		os << bichar_dim << std::endl;
		os << word_dim << std::endl;
		os << action_dim << std::endl;

		os << char_tune << std::endl;
		os << bichar_tune << std::endl;
		os << word_tune << std::endl;

		os << char_context << std::endl;
		os << char_repsentation_dim << std::endl;
		os << char_window_dim << std::endl;


		os << char_hidden_dim << std::endl;
		os << word_hidden_dim << std::endl;
		os << action_hidden_dim << std::endl;		

		os << char_lstm_dim << std::endl;
		os << word_lstm_dim << std::endl;
		os << action_lstm_dim << std::endl;
			
		os << app_hidden_dim << std::endl;
		os << sep_hidden_dim << std::endl;
	}


	void loadModel(std::ifstream& is){
		is >> beam ;
		is >> maxlength ;
		is >> action_num ;
		is >> delta ;

		int d_size;

		is >> d_size;
		
		for( int i = 0;i<d_size;i++){
			string temp; 
			is >> temp;
			dicts.insert(temp);
		}
		is >> nnRegular ;
		is >> adaAlpha ;
		is >> adaEps ;
		is >> dropProb ;

		is >> char_dim ;
		is >> chartype_dim ;
		is >> bichar_dim ;
		is >> word_dim ;
		is >> action_dim ;

		is >> char_tune ;
		is >> bichar_tune ;
		is >> word_tune ;

		is >> char_context ;
		is >> char_repsentation_dim ;
		is >> char_window_dim ;


		is >> char_hidden_dim ;
		is >> word_hidden_dim ;
		is >> action_hidden_dim ;		

		is >> char_lstm_dim ;
		is >> word_lstm_dim ;
		is >> action_lstm_dim ;
			
		is >> app_hidden_dim ;
		is >> sep_hidden_dim ;
	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */