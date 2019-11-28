package mains;

import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.semisupervised.GaussianFieldsByRandomWalkWithFriends;
import Classifier.supervised.modelAdaptation.Base;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithKmeans;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import clustering.KMeansAlg4Profile;
import structures._Corpus;

import java.io.IOException;
import java.util.HashMap;

public class HUBMain {

    public static void main(String[] args) throws IOException {

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        double trainRatio = 0;
        int displayLv = 1;
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        boolean enforceAdapt = true;
        String tokenModel = "./data/Model/en-token.bin"; // Token model.
        String prefix = "./data/CoLinAdapt";
        String data = "YelpNew";
        String featureSelection = "DF";

        /* specify the document analyzer and related parameters */
        int fv = 800;
        int fvSup = 5000;
        int lmTopK = 1000;
        double adaptRatio = 0.5;

        String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, data);
        String userFolder = String.format("%s/%s/Users", prefix, data);
        String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, data, fv);
        String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, data, fvSup);
        String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, data);
        String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, data, featureSelection, lmTopK);

//        /***Feature selection**/
//        MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, numberOfCores,true);
//        analyzer.LoadStopwords("./data/Model/stopwords.dat");
//        analyzer.loadUserDir(userFolder);
//
//        int minDF = 100;
//        int maxDF = 10000;
//        // we can only use DF for selecting features for language model
//        String lmFs = "DF";
//        // we can use different methods for selecting features for sentiment classification
//        String fs1 = "CHI";
//        String fs2 = "IG";
//        // top K features for sentiment classification
//        int topK = 5000;
//
//        // the line selects features for language model
//        analyzer.featureSelection(String.format("./data/%s_fv_lm_%s_%d.txt", data, lmFs, lmTopK), lmFs, maxDF, minDF, lmTopK);
//        // the line selects features for sentiment model
//        analyzer.featureSelection(String.format("./data/%s_fv_%s_%s_%d.txt", data, fs1, fs2, topK), fs1, fs2, maxDF, minDF, topK);

        if (fv == 5000 || fv == 3071) featureGroupFile = null;
        if (fvSup == 5000 || fv == 3071) featureGroupFileSup = null;
        if (lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;

        String friendFile = String.format("%s/%s/%sFriends.txt", prefix, data, data);
        // change the last variable to be true to use the new formated feature file
        Boolean newCV = false;
        MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, newCV);
        analyzer.config(trainRatio, adaptRatio, enforceAdapt);
        analyzer.loadUserDir(userFolder);
        analyzer.buildFriendship(friendFile);
        analyzer.setFeatureValues("TFIDF-sublinear", 0);
        HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

        double[] globalLM = analyzer.estimateGlobalLM();
        if (fv == 5000 || fv == 3071) featureGroupFile = null;
        if (fvSup == 5000 || fv == 3071) featureGroupFileSup = null;
        if (lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;

        /* specify the model and model related parameters */
        String model = "hub";
        double eta1 = 0.05;
        double eta2 = 0.05;
        double eta3 = 0.05;
        double eta4 = 0.05;
        double sdA = 0.0425;
        double sdB = 0.0425;
        int nuOfIterations = 2;
        int M = 6;
        int burnin = 10;
        int thinning = 3;

        // Concentration parameters
        double alpha = 0.01;
        double eta = 0.05;
        double beta = 0.01;

        // hub related parameters
        double rho = 0.05;
        boolean jointAll = false;

        if (model.equals("base")) {
            Base base = new Base(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
            base.loadUsers(analyzer.getUsers());
            base.setPersonalizedModel();
            base.test();
        } else if (model.equals("mtsvm")) {
            MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
            mtsvm.loadUsers(analyzer.getUsers());
            mtsvm.setBias(true);
            mtsvm.train();
            mtsvm.test();
        } else if (model.equals("mtclinkmeans")) {
            // kmeans over user weights learned from individual svms
            int kmeans = 100;
            int[] clusters;
            KMeansAlg4Profile alg = new KMeansAlg4Profile(classNumber, analyzer.getFeatureSize(), kmeans);
            alg.train(analyzer.getUsers());
            // The returned clusters contain the corresponding cluster index of each user.
            clusters = alg.assignClusters();
            CLinAdaptWithKmeans mtclinkmeans = new CLinAdaptWithKmeans(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, kmeans, clusters);
            // i, c, g represents the individual/cluster/global weight
            mtclinkmeans.setParameters(0, 1, 1);
            mtclinkmeans.loadUsers(analyzer.getUsers());
            mtclinkmeans.setDisplayLv(displayLv);
            mtclinkmeans.setLNormFlag(false);
            mtclinkmeans.train();
            mtclinkmeans.test();

        } else if (model.equals("gbssl")) {

            double learningRatio = 1.0;
            int k = 30, kPrime = 20; // k nearest labeled, k' nearest unlabeled
            double tAlpha = 1.0, tBeta = 0.1; // labeled data weight, unlabeled data weight
            double tDelta = 1e-5, tEta = 0.6; // convergence of random walk, weight of random walk
            boolean weightedAvg = true;

            String multipleLearner = "SVM";
            double C = 1.0;
            _Corpus c = analyzer.getCorpus();
            GaussianFieldsByRandomWalkWithFriends walk = new GaussianFieldsByRandomWalkWithFriends(c, multipleLearner, C,
                    learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg);
            walk.constructTrainTestDocs(analyzer.getUsers());
            walk.train();
            walk.test();
        } else {

            CLRWithDP adaptation = null;
            if (model.equals("mtclindp")) {
                adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup);
                ((MTCLinAdaptWithDP) adaptation).setsdB(sdB);
                ((MTCLinAdaptWithDP) adaptation).setR2TradeOffs(eta3, eta4);

            } else if (model.equals("mtclinhdp")) {
                adaptation = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
                ((MTCLinAdaptWithHDP) adaptation).loadLMFeatures(analyzer.getLMFeatures());
                ((MTCLinAdaptWithHDP) adaptation).setConcentrationParams(alpha, eta, beta);
                ((MTCLinAdaptWithHDP) adaptation).setsdB(sdB);
                ((MTCLinAdaptWithHDP) adaptation).setR2TradeOffs(eta3, eta4);

            } else if (model.equals("hub")) {
                adaptation = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
                ((MTCLinAdaptWithMMB) adaptation).loadLMFeatures(analyzer.getLMFeatures());
                ((MTCLinAdaptWithMMB) adaptation).setConcentrationParams(alpha, eta, beta);
                ((MTCLinAdaptWithMMB) adaptation).setsdB(sdB);
                ((MTCLinAdaptWithMMB) adaptation).setR2TradeOffs(eta3, eta4);
                ((MTCLinAdaptWithMMB) adaptation).setRho(rho);
                ((MTCLinAdaptWithMMB) adaptation).setJointSampling(jointAll);

            } else {
                System.out.println("[Error] The algorithm has not been developed yet!");
            }
            // commonly shared parameters.
            adaptation.setR1TradeOffs(eta1, eta2);
            adaptation.setM(M);
            adaptation.setsdA(sdA);
            adaptation.setBurnIn(burnin);
            adaptation.setThinning(thinning);
            adaptation.setNumberOfIterations(nuOfIterations);

            // training testing operations.
            adaptation.loadUsers(analyzer.getUsers());
            adaptation.setDisplayLv(displayLv);

            adaptation.train();
            adaptation.test();

            String perfFile = String.format("./data/%s_%s_performance_concise.txt", model, data);
            String perfDetailFile = String.format("./data/%s_%s_performance_detail.txt", model, data);

            // save user's performance in concise format, each user is one line
            // format: userID\tabF1 for negative class\tabF1 for positive class
            adaptation.savePerf(perfFile);

            /***save user's performance with details, each user has two lines:
             * the first line is the same as the output from previous function: userID\tabF1 for negative class\tabF1 for positive class
             * the second line prints the labels for all the reviews of the current user, each review is one triplet:
             * (each review's itemID, trueLabel, predicted Label)
             * e.g., (itemID_1, 1, 0)\tab(itemID_2, 1, 1)\tab()....
             ***/
            adaptation.savePerfWithDetail(perfDetailFile);

        }
    }
}
