package myMains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures.DPParameter;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.modelAdaptation.MMB.CLRWithMMB;
import Classifier.supervised.modelAdaptation.MMB.CLinAdaptWithMMB;
import Classifier.supervised.modelAdaptation.MMB.MTCLRWithMMB;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;

public class MyMMBExecution {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		DPParameter param = new DPParameter(args);

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		boolean enforceAdapt = true;
		String tokenModel = "./data/Model/en-token.bin"; // Token model.

//		String dataset = "YelpNew"; // "Amazon", "AmazonNew", "Yelp"
		String providedCV = String.format("%s/%s/SelectedVocab.csv", param.m_prefix, param.m_data); // CV.
		String userFolder = String.format("%s/%s/Users", param.m_prefix, param.m_data);
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fv);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fvSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", param.m_prefix, param.m_data);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", param.m_prefix, param.m_data, param.m_fs, param.m_lmTopK);
				
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends.txt", param.m_prefix,param.m_data,param.m_data);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, param.m_adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		double[] globalLM = analyzer.estimateGlobalLM();
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		CLRWithMMB adaptation = null;
		
		if(param.m_model.equals("mtmmb")){
			adaptation = new MTCLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
			adaptation.setQ(param.m_q);
			adaptation.setC(param.m_c);
		} else if(param.m_model.equals("clinmmb")){
			adaptation = new CLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, globalLM);
			((CLinAdaptWithMMB) adaptation).setsdB(param.m_sdB);
		} else if(param.m_model.equals("mtclinmmb")){
			adaptation = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
			((MTCLinAdaptWithMMB) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
			((CLinAdaptWithMMB) adaptation).setsdB(param.m_sdB);
		} else
			adaptation = new CLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);			
		
		// commonly shared parameters.
		adaptation.setR1TradeOffs(param.m_eta1, param.m_eta2);
		adaptation.setM(param.m_M);
		adaptation.setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
		adaptation.setsdA(param.m_sdA);
		
		adaptation.setRho(param.m_rho);
		adaptation.setBurnIn(param.m_burnin);
		adaptation.setThinning(param.m_thinning);
		adaptation.setNumberOfIterations(param.m_nuOfIterations);
		adaptation.setMultipleE(param.m_multipleE);
	
		adaptation.setJointSampling(param.m_jointAll);
		// training testing operations.
		adaptation.loadLMFeatures(analyzer.getLMFeatures());
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		long start = System.currentTimeMillis();

		if(param.m_trace){
			adaptation.trainTrace(param.m_data, start);
			adaptation.printEdgeCount(String.format("%s_edge_count_iter_%d_%d.txt", param.m_data, param.m_nuOfIterations, start));
			adaptation.test();
		} else{
			adaptation.train();
			adaptation.test();
		}
		// record the time
		long end = System.currentTimeMillis();
		System.out.println("\n[Info]Start time: " + start);
		System.out.println("[Info]End time: " + end);
		// the total time of training and testing in the unit of hours
		double hours = (end - start)/(1000*60*60);
		System.out.print(String.format("[Time]This training+testing process took %.2f hours.\n", hours));
		if(param.m_saveModel){
			String dir = String.format("%s/%d_%s", param.m_saveDir, end, param.m_data);
			((MTCLinAdaptWithMMB) adaptation).saveEverything(dir);
		}
		
	}
}
