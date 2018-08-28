package structures;

import java.util.HashMap;

public class _Review extends _Doc {

	public enum rType {
		TRAIN, // for training the global model
		ADAPTATION, // for training the personalized model
		TEST, // for testing
		SEPARATE // added by Lin for sanity check.
	}
	
	String m_userID;
	String m_category; 
	rType m_type; // specification of this review
	
	//Constructor for route project.
	public _Review(int ID, String source, int ylabel){
		super(ID, source, ylabel);
	}
	
	public _Review(int ID, String source, int ylabel, String userID, String productID, String category, long timeStamp){
		super(ID, source, ylabel);
		m_userID = userID;
		m_itemID = productID;
		m_category = category;
		m_timeStamp = timeStamp;
		m_type = rType.TRAIN; // by default, every review is used for training the global model
	}
	
	public rType getType() {
		return m_type;
	}
	
	public void setType(rType type) {
		m_type = type;
	}
	
	//Compare the timestamp of two documents and sort them based on timestamps.
	@Override
	public int compareTo(_Doc d){
		if(m_timeStamp < d.m_timeStamp)
			return -1;
		else if(m_timeStamp == d.m_timeStamp)
			return 0;
		else 
			return 1;
	}

	//Access the userID of the review.
	public String getUserID(){
		return m_userID;
	}
	
	@Override
	public String toString() {
		return String.format("%s-%s-%s-%s", m_userID, m_itemID, m_category, m_type);
	}
	
	// Added by Lin for experimental purpose.
	public String getCategory(){
		return m_category;
	}
	
	// Added for the HDP algorithm.
	_HDPThetaStar m_hdpThetaStar;
	public void setHDPThetaStar(_HDPThetaStar s){
		m_hdpThetaStar = s;
	}
	
	// key: hdpThetaStar, value: count
	HashMap<_HDPThetaStar, Integer> m_thetaCountMap = new HashMap<_HDPThetaStar, Integer>();
	
	// Increase the current hdpThetaStar count.
	public void updateThetaCountMap(int c){
		if(!m_thetaCountMap.containsKey(m_hdpThetaStar)){
			m_thetaCountMap.put(m_hdpThetaStar, c);
		} else{
			int v = m_thetaCountMap.get(m_hdpThetaStar);
			m_thetaCountMap.put(m_hdpThetaStar, v+c);
		}
	}
	
	public HashMap<_HDPThetaStar, Integer> getThetaCountMap(){
		return m_thetaCountMap;
	}
	public void clearThetaCountMap(){
		m_thetaCountMap.clear();
	}
	public _HDPThetaStar getHDPThetaStar(){
		return m_hdpThetaStar;
	}
	
	// Added by Lin for HDP evaluation.
	double[] m_cluPosterior;
	
	public void setClusterPosterior(double[] posterior) {
		if (m_cluPosterior==null || m_cluPosterior.length != posterior.length)
			m_cluPosterior = new double[posterior.length];
		System.arraycopy(posterior, 0, m_cluPosterior, 0, posterior.length);
	}
	
	public double[] getCluPosterior(){
		return m_cluPosterior;
	}
	double m_L4New = 0;
	public void setL4NewCluster(double l){
		m_L4New = l;
	}
	
	public double getL4NewCluster(){
		return m_L4New;
	}
	// total count of words of the review represented in language model.
	double m_lmSum = -1;
	public double getLMSum(){
		if(m_lmSum != -1)
			return m_lmSum;
		else{
			double sum = 0;
			for(_SparseFeature sf: m_lm_x_sparse)
				sum += sf.getValue();
			return sum;
		}
	}
	// Assign each review a confidence, used in hdp model.
	protected double m_confidence = 1;
	public void setConfidence(double conf){
		m_confidence = conf;
	}
	
	public double getConfidence(){
		return m_confidence;
	}
}
