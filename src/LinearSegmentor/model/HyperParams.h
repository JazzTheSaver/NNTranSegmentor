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

public:
	HyperParams(){
		maxlength = max_sentence_clength + 1;
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		//please specify diction outside
		beam = opt.beam;
		delta = opt.delta;
		bAssigned = true;

		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		dropProb = opt.dropProb;
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
		std::cout << "beam = " << beam << std::endl;
		std::cout << "maxlength = " << maxlength << std::endl;
		std::cout << "action_num = " << action_num << std::endl;
		std::cout << "delta = " << delta << std::endl;


		std::cout << "nnRegular = " << nnRegular << std::endl;
		std::cout << "adaAlpha = " << adaAlpha << std::endl;
		std::cout << "adaEps = " << adaEps << std::endl;
		std::cout << "dropProb = " << dropProb << std::endl;

		std::cout << "=======================" << std::endl;

	}

	void saveModel(std::ofstream& os) {
		os << beam << std::endl;
		os << maxlength << std::endl;
		os << action_num << std::endl;
		os << delta << std::endl;

		os << dicts.size() << std::endl;
		for (unordered_set<string>::iterator it = dicts.begin(); it != dicts.end(); it++) {
			os << *it << std::endl;
		}

		os << nnRegular << std::endl;
		os << adaAlpha << std::endl;
		os << adaEps << std::endl;
		os << dropProb << std::endl;
	}void loadModel(std::ifstream& is) {
		is >> beam;
		is >> maxlength;
		is >> action_num;
		is >> delta;

		int d_size;

		is >> d_size;
		for (int i = 0; i<d_size; i++) {
			string temp;
			is >> temp;
			dicts.insert(temp);
		}
		is >> nnRegular;
		is >> adaAlpha;
		is >> adaEps;
		is >> dropProb;

	}
private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */