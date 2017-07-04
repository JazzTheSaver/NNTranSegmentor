#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	//neural parameters
	Alphabet embeded_words; // words
	LookupTable word_table; // should be initialized outside	
	Alphabet embeded_chars; // chars
	LookupTable char_table; // should be initialized outside	
	Alphabet embeded_chartypes; // chars
	LookupTable chartype_table; // should be initialized outside	
	Alphabet embeded_bichars; // bichars
	LookupTable bichar_table; // should be initialized outside
	Alphabet embeded_actions; // bichars
	LookupTable action_table; // should be initialized outside
	
	UniParams char_tanh_conv; // hidden
	LSTM1Params char_left_lstm; //left lstm
	LSTM1Params char_right_lstm; //right lstm
	BiParams word_conv;
	LSTM1Params word_lstm;
	BiParams action_conv;
	LSTM1Params action_lstm;
	FourParams sep_hidden;
	TriParams app_hidden;
	UniParams sep_score;
	UniParams app_score;
	
	//should be initialized outside
	Alphabet words; // words
	Alphabet chars; // chars
	Alphabet charTypes; // char type
	


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem){
		// some model parameters should be initialized outside
		if (words.size() <= 0 || chars.size() <= 0){
			return false;
		}
		//neural features
		char_tanh_conv.initial(opts.char_hidden_dim, opts.char_window_dim, true, mem);
		char_left_lstm.initial(opts.char_lstm_dim, opts.char_hidden_dim, mem); //left lstm
		char_right_lstm.initial(opts.char_lstm_dim, opts.char_hidden_dim, mem); //right lstm
		word_conv.initial(opts.word_hidden_dim, opts.word_dim, opts.word_dim, true, mem);
		word_lstm.initial(opts.word_lstm_dim, opts.word_hidden_dim, mem);
		action_conv.initial(opts.action_hidden_dim, opts.action_dim, opts.action_dim, true, mem);
		action_lstm.initial(opts.action_lstm_dim, opts.action_hidden_dim, mem);
		sep_hidden.initial(opts.sep_hidden_dim, opts.char_lstm_dim, opts.char_lstm_dim, opts.word_lstm_dim, opts.action_lstm_dim, true, mem);
		app_hidden.initial(opts.app_hidden_dim, opts.char_lstm_dim, opts.char_lstm_dim, opts.action_lstm_dim, true, mem);
		sep_score.initial(1, opts.sep_hidden_dim, false, mem);
		app_score.initial(1, opts.app_hidden_dim, false, mem);

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		//neural features
		word_table.exportAdaParams(ada);
		char_table.exportAdaParams(ada);
		chartype_table.exportAdaParams(ada);
		bichar_table.exportAdaParams(ada);
		action_table.exportAdaParams(ada);

		char_tanh_conv.exportAdaParams(ada);
		char_left_lstm.exportAdaParams(ada); 
		char_right_lstm.exportAdaParams(ada);
		word_conv.exportAdaParams(ada);
		word_lstm.exportAdaParams(ada);
		action_conv.exportAdaParams(ada);
		action_lstm.exportAdaParams(ada);
		sep_hidden.exportAdaParams(ada);
		app_hidden.exportAdaParams(ada);
		sep_score.exportAdaParams(ada);
		app_score.exportAdaParams(ada);

	}


	// will add it later
	void saveModel(std::ofstream& os) {
		embeded_words.write(os);
		word_table.save(os);
		embeded_chars.write(os);
		char_table.save(os);
		embeded_chartypes.write(os);
		chartype_table.save(os);
		embeded_bichars.write(os);
		bichar_table.save(os);
		embeded_actions.write(os);
		action_table.save(os);

		char_tanh_conv.save(os);
		char_left_lstm.save(os);
		char_right_lstm.save(os);
		word_conv.save(os);
		word_lstm.save(os);
		action_conv.save(os);
		action_lstm.save(os);
		sep_hidden.save(os);
		app_hidden.save(os);
		sep_score.save(os);
		app_score.save(os);

		words.write(os);
		chars.write(os);
		charTypes.write(os);


	}

	void loadModel(std::ifstream& is, AlignedMemoryPool *mem = NULL) {
		embeded_words.read(is);
		word_table.load(is, &embeded_words, mem);
		embeded_chars.read(is);
		char_table.load(is, &embeded_chars, mem);
		embeded_chartypes.read(is);
		chartype_table.load(is, &embeded_chartypes, mem);
		embeded_bichars.read(is);
		bichar_table.load(is, &embeded_bichars, mem);
		embeded_actions.read(is);
		action_table.load(is, &embeded_actions, mem);

		char_tanh_conv.load(is);
		char_left_lstm.load(is);
		char_right_lstm.load(is);
		word_conv.load(is);
		word_lstm.load(is);
		action_conv.load(is);
		action_lstm.load(is);
		sep_hidden.load(is);
		app_hidden.load(is);
		sep_score.load(is);
		app_score.load(is);


		words.read(is);
		chars.read(is);
		charTypes.read(is);
	}

};

#endif /* SRC_ModelParams_H_ */