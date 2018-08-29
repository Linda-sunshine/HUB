package mains;

import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.semisupervised.GaussianFieldsByRandomWalkWithFriends;
import Classifier.supervised.modelAdaptation.Base;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithKmeans;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import clustering.KMeansAlg4Profile;
import structures.DPParameter;
import structures._Corpus;

import java.io.IOException;
import java.util.HashMap;

public class HUBExecution {

    public static void main(String[] args) throws IOException {
        DPParameter param = new DPParameter(args);

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        double trainRatio = 0;
        int displayLv = 1;
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        boolean enforceAdapt = true;
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String providedCV = String.format("%s/%s/SelectedVocab.csv", param.m_prefix, param.m_data); // CV.
        String userFolder = String.format("%s/%s/Users", param.m_prefix, param.m_data);
        String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fv);
        String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fvSup);
        String globalModel = String.format("%s/%s/GlobalWeights.txt", param.m_prefix, param.m_data);
        String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", param.m_prefix, param.m_data, param.m_fs, param.m_lmTopK);

        if (param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
        if (param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
        if (param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;

        String friendFile = String.format("%s/%s/%sFriends.txt", param.m_prefix, param.m_data, param.m_data);
        MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
        analyzer.config(trainRatio, param.m_adaptRatio, enforceAdapt);
        analyzer.loadUserDir(userFolder);
        analyzer.buildFriendship(friendFile);
        analyzer.setFeatureValues("TFIDF-sublinear", 0);
        HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

        double[] globalLM = analyzer.estimateGlobalLM();
        if (param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
        if (param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
        if (param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;

        if (param.m_model.equals("base")) {
            Base base = new Base(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
            base.loadUsers(analyzer.getUsers());
            base.setPersonalizedModel();
            base.test();
        } else if (param.m_model.equals("mtsvm")) {
            MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
            mtsvm.loadUsers(analyzer.getUsers());
            mtsvm.setBias(true);
            mtsvm.train();
            mtsvm.test();
        } else if (param.m_model.equals("mtclinkmeans")) {
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

        } else if (param.m_model.equals("gbssl")) {

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
            if (param.m_model.equals("mtclindp")) {
                adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup);
                ((MTCLinAdaptWithDP) adaptation).setsdB(param.m_sdB);
                ((MTCLinAdaptWithDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);

            } else if (param.m_model.equals("mtclinhdp")) {
                adaptation = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
                ((MTCLinAdaptWithHDP) adaptation).loadLMFeatures(analyzer.getLMFeatures());
                ((MTCLinAdaptWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
                ((MTCLinAdaptWithHDP) adaptation).setsdB(param.m_sdB);
                ((MTCLinAdaptWithHDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);

            } else if (param.m_model.equals("hub")) {
                adaptation = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
                ((MTCLinAdaptWithMMB) adaptation).loadLMFeatures(analyzer.getLMFeatures());
                ((MTCLinAdaptWithMMB) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
                ((MTCLinAdaptWithMMB) adaptation).setsdB(param.m_sdB);
                ((MTCLinAdaptWithMMB) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
                ((MTCLinAdaptWithMMB) adaptation).setRho(param.m_rho);
                ((MTCLinAdaptWithMMB) adaptation).setJointSampling(param.m_jointAll);

            } else {
                System.out.println("[Error] The algorithm has not been developed yet!");
            }
            // commonly shared parameters.
            adaptation.setR1TradeOffs(param.m_eta1, param.m_eta2);
            adaptation.setM(param.m_M);
            adaptation.setsdA(param.m_sdA);
            adaptation.setBurnIn(param.m_burnin);
            adaptation.setThinning(param.m_thinning);
            adaptation.setNumberOfIterations(param.m_nuOfIterations);

            // training testing operations.
            adaptation.loadUsers(analyzer.getUsers());
            adaptation.setDisplayLv(displayLv);

            adaptation.train();
            adaptation.test();
        }
    }
}
