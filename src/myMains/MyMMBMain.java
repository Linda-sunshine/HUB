package myMains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;


public class MyMMBMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
		boolean enforceAdapt = true;
 
		String dataset = "Amazon"; // "Amazon", "YelpNew"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		int lmTopK = 1000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;

		String fs = "DF";//"IG_CHI"
		
		String prefix = "./data/CoLinAdapt";

		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String userFolder = String.format("%s/%s/Users", prefix, dataset);
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

		// best parameter for yelp so far.
		double[] globalLM = analyzer.estimateGlobalLM();
		double alpha = 0.1, eta = 0.05, beta = 0.01;
		double sdA = 0.0425, sdB = 0.0425;

		MTCLinAdaptWithMMB mmb = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		mmb.setR2TradeOffs(eta3, eta4);
		mmb.setsdA(sdA);
		mmb.setsdB(sdB);
		mmb.setR1TradeOffs(eta1, eta2);
		mmb.setConcentrationParams(alpha, eta, beta);
		
		double rho = 0.2; 
		int burnin = 10, iter = 30, thin = 3;
		boolean jointAll = false;
		mmb.setRho(rho);
		mmb.setBurnIn(burnin);
		mmb.setThinning(thin);// default 3
		mmb.setNumberOfIterations(iter);
		
		mmb.setJointSampling(jointAll);
		mmb.loadLMFeatures(analyzer.getLMFeatures());
		mmb.loadUsers(analyzer.getUsers());
		mmb.setDisplayLv(displayLv);
		long start = System.currentTimeMillis();

		boolean trace = true; 
		if(trace){
			iter = 200; thin = 1; burnin = 0;
			mmb.setNumberOfIterations(iter);
			mmb.setThinning(thin);
			mmb.setBurnIn(burnin);
			mmb.trainTrace(dataset, start);
		} else{
			mmb.train();
			mmb.test(); 
		}
		long end = System.currentTimeMillis();
		System.out.println("\n[Info]Start time: " + start);
		System.out.println("[Info]End time: " + end);
		// the total time of training and testing in the unit of hours
		double hours = (end - start)/(1000*60);
		System.out.print(String.format("[Time]This training+testing process took %.2f mins.\n", hours));
	}
}
