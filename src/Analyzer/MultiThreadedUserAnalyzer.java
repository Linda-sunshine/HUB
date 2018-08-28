package Analyzer;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import structures.TokenizeResult;
import structures._Doc;
import structures._Review;
import structures._Review.rType;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * @author Mohammad Al Boni
 * Multi-threaded extension of UserAnalyzer
 */
public class MultiThreadedUserAnalyzer extends UserAnalyzer {

	protected int m_numberOfCores;
	protected Tokenizer[] m_tokenizerPool;
	protected SnowballStemmer[] m_stemmerPool;
	protected Object m_allocReviewLock=null;
	protected Object m_corpusLock=null;
	protected Object m_rollbackLock=null;
	private Object m_featureStatLock=null;
	private Object m_mapLock = null;
	
	public MultiThreadedUserAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
					throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, b);
		
		m_numberOfCores = numberOfCores;
		
		// since DocAnalyzer already contains a tokenizer, then we can user it and define a pool with length of m_numberOfCores - 1
		m_tokenizerPool = new Tokenizer[m_numberOfCores-1]; 
		m_stemmerPool = new SnowballStemmer[m_numberOfCores-1];
		for(int i=0;i<m_numberOfCores-1;++i){
			m_tokenizerPool[i] = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
			m_stemmerPool[i] = new englishStemmer();
		}
		
		m_allocReviewLock = new Object();// lock when collecting review statistics
		m_corpusLock = new Object(); // lock when collecting class statistics 
		m_rollbackLock = new Object(); // lock when revising corpus statistics
		m_featureStatLock = new Object();
		m_mapLock = new Object();
	}
	
	//Load all the users.
	@Override
	public void loadUserDir(String folder){
		if(folder == null || folder.isEmpty())
			return;

		File dir = new File(folder);
		final File[] files=dir.listFiles();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<m_numberOfCores;++i){
			threads.add(  (new Thread() {
				int core;
				@Override
				public void run() {
					try {
						for (int j = 0; j + core <files.length; j += m_numberOfCores) {
							File f = files[j+core];
							// && f.getAbsolutePath().endsWith("txt")
							if(f.isFile()){//load the user								
								loadUser(f.getAbsolutePath(),core);
							}
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core ) {
					this.core = core;
					return this;
				}
			}).initialize(i));
			
			threads.get(i).start();
		}
		for(int i=0;i<m_numberOfCores;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		} 
		
		// process sub-directories
		int count=0;
		for(File f:files ) 
			if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
			else
				count++;

		System.out.format("\n%d users are loaded from %s...\n", count, folder);
	}
	
		
	// Load one file as a user here. 
	protected void loadUser(String filename, int core){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.				
			
			// Skip the first line since it is user name.
			reader.readLine(); 

			String productID, source, category="";
			ArrayList<_Review> reviews = new ArrayList<_Review>();

			_Review review;
			int ylabel;
			long timestamp=0;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());
							
				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, productID, category, timestamp);
					if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
						reviews.add(review);
					}
				}
			}
			if(reviews.size() > 1){//at least one for adaptation and one for testing
				synchronized (m_allocReviewLock) {
					allocateReviews(reviews);	
					m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
				}
			} else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
				review = reviews.get(0);
				synchronized (m_rollbackLock) {
					rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
				}
			}

			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	

	
	//Tokenizing input text string
	private String[] Tokenizer(String source, int core){
		String[] tokens = getTokenizer(core).tokenize(source);
		return tokens;
	}
	
	//Snowball Stemmer.
	private String SnowballStemming(String token, int core){
		SnowballStemmer stemmer = getStemmer(core);
		stemmer.setCurrent(token);
		if(stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected TokenizeResult TokenizerNormalizeStemmer(String source, int core){
		String[] tokens = Tokenizer(source, core); //Original tokens.
		TokenizeResult result = new TokenizeResult(tokens);

		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]),core);
		
		LinkedList<String> Ngrams = new LinkedList<String>();
		int tokenLength = tokens.length, N = m_Ngram;			

		for(int i=0; i<tokenLength; i++) {
			String token = tokens[i];
			boolean legit = isLegit(token);
			if (legit) 
				Ngrams.add(token);//unigram
			else
				result.incStopwords();

			//N to 2 grams
			if (!isBoundary(token)) {
				for(int j=i-1; j>=Math.max(0, i-N+1); j--) {	
					if (isBoundary(tokens[j]))
						break;//touch the boundary

					token = tokens[j] + "-" + token;
					legit |= isLegit(tokens[j]);
					if (legit)//at least one of them is legitimate
						Ngrams.add(token);
				}
			}
		}

		result.setTokens(Ngrams.toArray(new String[Ngrams.size()]));
		return result;
	}
	
	/*Analyze a document and add the analyzed document back to corpus.*/
	protected boolean AnalyzeDoc(_Doc doc, int core) {
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource(),core);// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();

		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		
		if (spVct.size()>m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			doc.setStopwordProportion(result.getStopwordProportion());
			synchronized (m_corpusLock) {
				m_corpus.addDoc(doc);
				m_classMemberNo[y]++;
			}
			if (m_releaseContent)
				doc.clearSource();
			
			return true;
		} else {
			/****Roll back here!!******/
			synchronized (m_rollbackLock) {
				rollBack(spVct, y);
			}
			return false;
		}
	}
	
	//convert the input token sequence into a sparse vector (docWordMap cannot be changed)
	// Since multiple threads access the featureStat, we need lock for this variable.
	@Override
	protected HashMap<Integer, Double> constructSpVct(String[] tokens, int y, HashMap<Integer, Double> docWordMap) {
		int index = 0;
		double value = 0;
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		
		for (String token : tokens) {//tokens could come from a sentence or a document
			// CV is not loaded, take all the tokens as features.
			if (!m_isCVLoaded) {
				if (m_featureNameIndex.containsKey(token)) {
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						if (docWordMap==null || !docWordMap.containsKey(index)) {
							if(m_featureStat.containsKey(0)){
								synchronized(m_featureStatLock){
									m_featureStat.get(token).addOneDF(y);
								}
							}
						}
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					if(m_featureStat.containsKey(token)){
						synchronized(m_featureStatLock){
							m_featureStat.get(token).addOneDF(y);
						}
					}
				}
				if(m_featureStat.containsKey(token)){
					synchronized(m_featureStatLock){
						m_featureStat.get(token).addOneTTF(y);
					}
				}
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					if (!m_isCVStatLoaded && (docWordMap==null || !docWordMap.containsKey(index))){
						synchronized(m_featureStatLock){
							m_featureStat.get(token).addOneDF(y);
						}
					}
				}
				
				if (!m_isCVStatLoaded){
					synchronized(m_featureStatLock){
						m_featureStat.get(token).addOneTTF(y);
					}
				}
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		return spVct;
	}
	
	
	// return a tokenizer using the core number
	private Tokenizer getTokenizer(int index){
		if(index==m_numberOfCores-1)
			return m_tokenizer;
		else
			return m_tokenizerPool[index];
	}
	
	// return a stemmer using the core number
	private SnowballStemmer getStemmer(int index){
		if(index==m_numberOfCores-1)
			return m_stemmer;
		else
			return m_stemmerPool[index];
	}

	protected HashMap<String, Integer> m_userIDIndex;
	// Added by Lin. Load user weights from learned models to construct neighborhood.
	public void loadUserWeights(String folder, String suffix){
		if(folder == null || folder.isEmpty())
			return;
		String userID;
		int userIndex, count = 0;
		double[] weights;
		constructUserIDIndex();
		File dir = new File(folder);
		
		if(!dir.exists()){
			System.err.print("[Info]Directory doesn't exist!");
		} else{
			for(File f: dir.listFiles()){
				if(f.isFile() && f.getName().endsWith(suffix)){
					int endIndex = f.getName().lastIndexOf(".");
					userID = f.getName().substring(0, endIndex);
					if(m_userIDIndex.containsKey(userID)){
						userIndex = m_userIDIndex.get(userID);
						weights = loadOneUserWeight(f.getAbsolutePath());
						m_users.get(userIndex).setSVMWeights(weights);
						count++;
					}
				}
			}
		}
		System.out.format("%d users weights are loaded!\n", count);
	}
	public void constructUserIDIndex(){
		m_userIDIndex = new HashMap<String, Integer>();
		for(int i=0; i<m_users.size(); i++)
			m_userIDIndex.put(m_users.get(i).getUserID(), i);
	}
	
	// Added by Lin. Load one user's weights.
	public double[] loadOneUserWeight(String fileName) {
		double[] weights = new double[getFeatureSize()];
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(fileName), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				String[] ws = line.split(",");
				if (ws.length != getFeatureSize() + 1)
					System.out.println("[error]Wrong dimension of the user's weights!");
				else {
					weights = new double[ws.length];
					for (int i = 0; i < ws.length; i++) {
						weights[i] = Double.valueOf(ws[i]);
					}
				}
			}
			reader.close();
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", fileName);
			e.printStackTrace();
		}
		return weights;
	}
	// assign some of the users for testing only.
	public void separateUsers(int k){
		int count = 0;
		double light = 0, medium = 0;
		while(count < k){
			if(m_users.get(count).getReviewSize() <= 10)
				light++;
			else if(m_users.get(count).getReviewSize() <= 50)
				medium++;
			for(_Review r: m_users.get(count++).getReviews()){
				if(r.getType() == rType.ADAPTATION){
					m_adaptSize--;
					r.setType(rType.TEST);
					m_testSize++;
					if(r.getYLabel() == 1){
						m_pCount[1]--;
						m_pCount[2]++;
					}
				}
			}
		}
		System.out.print(String.format("[Prob Info]Light: %.4f, medium: %.4f, heavy: %.4f\n", light/k, medium/k, (k-light-medium)/k));
	}
	
	/** Construct user network for analysis****/
	// key: user id; value: friends array.
	HashMap<String, String[]> m_trainMap = new HashMap<String, String[]>();
	HashMap<String, String[]> m_testMap = new HashMap<String, String[]>();

	public void buildFriendship(String filename){
		try{
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String[] users, friends;
			while((line = reader.readLine()) != null){
				users = line.trim().split("\t");
				friends = Arrays.copyOfRange(users, 1, users.length);
				if(friends.length == 0){
					continue;
				}
				m_trainMap.put(users[0], friends);
			}
			reader.close();
			System.out.format("%d users have friends!", m_trainMap.size());
			// map friends to users.
			int count = 0;
			for(_User u: m_users){
				if(m_trainMap.containsKey(u.getUserID())){
					count++;
					u.setFriends(m_trainMap.get(u.getUserID()));
				}
			}
			System.out.format("%d users' friends are set!\n", count);
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	public void buildNonFriendship(String filename){
		
		System.out.println("[Info]Non-friendship file is loaded from " + filename);
		HashMap<String, String[]> nonFriendMap = new HashMap<String, String[]>();

		try{
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String[] users, nonFriends;
			while((line = reader.readLine()) != null){
				users = line.trim().split("\t");
				nonFriends = Arrays.copyOfRange(users, 1, users.length);
				if(nonFriends.length == 0){
					continue;
				}
				nonFriendMap.put(users[0], nonFriends);
			}
			reader.close();
			System.out.format("%d users have non-friends!\n", nonFriendMap.size());
			// map friends to users.
			int count = 0;
			for(_User u: m_users){
				if(nonFriendMap.containsKey(u.getUserID())){
					count++;
					u.setNonFriends(nonFriendMap.get(u.getUserID()));
				}
			}
			System.out.format("%d users' non-friends are set!\n", count);
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	// load a friendship file 
	public HashMap<String, String[]> loadFriendFile(String filename){
		HashMap<String, String[]> neighborsMap = new HashMap<String, String[]>();
		try{
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String[] users, friends;
			while((line = reader.readLine()) != null){
				users = line.trim().split("\t");
				friends = Arrays.copyOfRange(users, 1, users.length);
				if(friends.length == 0){
					continue;
				}
				neighborsMap.put(users[0], friends);
			}
			reader.close();
			System.out.format("%d users' friends are loaded!\n", neighborsMap.size());
		} catch(IOException e){
			e.printStackTrace();
		}
		return neighborsMap;
	}
	
	// load the test user friends, for link prediction only
	public void loadTestFriendship(String filename){
		try{
			m_testMap.clear();
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String[] users, friends;
			while((line = reader.readLine()) != null){
				users = line.trim().split("\t");
				friends = Arrays.copyOfRange(users, 1, users.length);
				if(friends.length == 0){
					continue;
				}
				m_testMap.put(users[0], friends);
			}
			reader.close();
			// map friends to users.
			for(_User u: m_users){
				if(m_testMap.containsKey(u.getUserID()))
					u.setTestFriends(m_testMap.get(u.getUserID()));
			}
			checkFriendSize();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public HashMap<String, String[]>  getTrainMap(){
		return m_trainMap;
	}
	
	public HashMap<String, String[]> getTestMap(){
		return m_testMap;
	}
	
	// save the user-user pairs to graphlab for model training.
	public void saveUserUserPairs(String dir){
		int trainUser = 0, testUser = 0, trainPair = 0, testPair = 0;
		try{
			PrintWriter trainWriter = new PrintWriter(new File(dir+"train.csv"));
			PrintWriter testWriter = new PrintWriter(new File(dir+"test.csv"));
			trainWriter.write("user_id,item_id,rating\n");
			testWriter.write("user_id,item_id,rating\n");
			for(_User u: m_users){
				if(u.getFriendSize() != 0){                
					trainUser++;
					for(String frd: u.getFriends()){
						trainPair++;
						trainWriter.write(String.format("%s,%s,%d\n", u.getUserID(), frd, 1));
						trainWriter.write(String.format("%s,%s,%d\n", frd, u.getUserID(), 1));

					}
				}
				// for test users, we also need to write out non-friends
				if(u.getTestFriendSize() != 0){
					testUser++;
					for(_User nei: m_users){
						String neiID = nei.getUserID();
						if(u.hasFriend(neiID) || u.getUserID().equals(neiID))
							continue;
						else if(u.hasTestFriend(neiID)){
							testPair++;
							testWriter.write(String.format("%s,%s,%d\n", u.getUserID(), neiID, 1));
							testWriter.write(String.format("%s,%s,%d\n", neiID, u.getUserID(), 1));
						} else if(m_trainMap.containsKey(neiID)){
							testPair++;
							testWriter.write(String.format("%s,%s,%d\n", u.getUserID(), neiID, 0));
							testWriter.write(String.format("%s,%s,%d\n", neiID, u.getUserID(), 0));
						}
					}
				}
			}
			trainWriter.close();
			testWriter.close();
			System.out.format("[Info]Finish writing (%d,%d) training users/pairs, (%d,%d) testing users/pairs.\n", trainUser, trainPair, testUser, testPair);
		} catch(IOException e){
			e.printStackTrace();
		}
		
	}
	public void checkFriendSize(){
		int train = 0, test = 0;
		for(_User u: m_users){
			if(u.getFriendSize() != 0)
				train++;
			if(u.getTestFriendSize() != 0)
				test++;
		}
		System.out.format("[Check]%d users have train friends, %d users have test friends.\n", train, test);
	}
	
	// filter the friends who are not in the list and return a neat hashmap
	public HashMap<String, ArrayList<String>> filterFriends(HashMap<String, String[]> neighborsMap){
		double sum = 0;
		HashMap<String, _User> userMap = new HashMap<String, _User>();
		for(_User u: m_users){
			userMap.put(u.getUserID(), u);
		}
		HashMap<String, ArrayList<String>> frdMap = new HashMap<String, ArrayList<String>>();
		for(String uid: neighborsMap.keySet()){
			if(!userMap.containsKey(uid)){
				System.out.println("The user does not exist in user set!");
				continue;
			}
			ArrayList<String> frds = new ArrayList<>();
			for(String frd: neighborsMap.get(uid)){
				if(!neighborsMap.containsKey(frd))
					continue;
				if(contains(neighborsMap.get(frd), uid)){
					frds.add(frd);
				} else {
					System.out.println("asymmetric");
				}
			}
			if(frds.size() > 0){
				frdMap.put(uid, frds);
				sum += frds.size();
			}
		}
		System.out.format("%d users' friends are recorded, avg friends: %.2f.\n", frdMap.size(), sum/frdMap.size());
		return frdMap;
	}
	public boolean contains(String[] strs, String str){
		if(strs == null || strs.length == 0)
			return false;
		for(String s: strs){
			if(str.equals(s))
				return true;
		}
		return false;
	}

	public void writeFriends(String filename, HashMap<String, ArrayList<String>> frdMap){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(String uid: frdMap.keySet()){
				writer.write(uid+"\t");
				ArrayList<String> frds = frdMap.get(uid);
				for(int i=0; i<frds.size(); i++){
					if(i != frds.size()-1){
						writer.write(frds.get(i)+"\t");
					} else
						writer.write(frds.get(i)+"\n");
				}
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public boolean hasFriend(String[] frds, String frd){
		for(String f: frds){
			if(f.equals(frd))
				return true;
		}
		return false;
	}
	
	public void rmMultipleReviews4OneItem(){
		Set<String> items = new HashSet<String>();
		ArrayList<Integer> indexes = new ArrayList<Integer>();
		int uCount = 0, rCount = 0;
		boolean flag = false;
		for(_User u: m_users){
			ArrayList<_Review> reviews = u.getReviews();
			items.clear();
			indexes.clear();
			for(int i=0; i<reviews.size(); i++){
				_Review r = reviews.get(i);
				if(items.contains(r.getItemID())){
					indexes.add(i);
					rCount++;
					flag = true;
				} else {
					items.add(r.getItemID());
				}
			}
			// record the user number
			if(flag){
				uCount++;
				flag = false;
			}
			// remove the reviews.
			Collections.sort(indexes, Collections.reverseOrder());
			for(int idx: indexes){
				reviews.remove(idx);
			}
			u.constructTrainTestReviews();
		}
		System.out.format("%d users have %d duplicate reviews for items.\n", uCount, rCount);
	}
	
	
	/***
	 * The following codes are used in cf for ETBIR.
	 */
	HashMap<String, _User> m_userMap = new HashMap<String, _User>();
	//Load users' test reviews.
	public void loadTestUserDir(String folder){
		
		// construct the training user map first
		for(_User u: m_users){
			if(!m_userMap.containsKey(u.getUserID()))
				m_userMap.put(u.getUserID(), u);
			else
				System.err.println("[error] The user already exists in map!!");
		}
		
		if(folder == null || folder.isEmpty())
			return;

		File dir = new File(folder);
		final File[] files=dir.listFiles();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<m_numberOfCores;++i){
			threads.add(  (new Thread() {
				int core;
				@Override
				public void run() {
					try {
						for (int j = 0; j + core <files.length; j += m_numberOfCores) {
							File f = files[j+core];
							// && f.getAbsolutePath().endsWith("txt")
							if(f.isFile()){//load the user								
								loadTestUserReview(f.getAbsolutePath(),core);
							}
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
						
				private Thread initialize(int core ) {
					this.core = core;
					return this;
				}
			}).initialize(i));
					
			threads.get(i).start();
		}
		for(int i=0;i<m_numberOfCores;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		} 
				
		// process sub-directories
		int count=0;
		for(File f:files ) 
			if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
			else
				count++;

		System.out.format("%d users are loaded from %s...\n", count, folder);
	}

	
	// Load one file as a user here. 
	protected void loadTestUserReview(String filename, int core){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.				
			
			// Skip the first line since it is user name.
			reader.readLine(); 

			String productID, source, category="";
			ArrayList<_Review> reviews = new ArrayList<_Review>();

			_Review review;
			int ylabel;
			long timestamp=0;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());
							
				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, productID, category, timestamp);
					if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
						reviews.add(review);
					}
				}
			}
			if(reviews.size() > 1){//at least one for adaptation and one for testing
				synchronized (m_allocReviewLock) {
//					allocateReviews(reviews);	
					if(m_userMap.containsKey(userID)){
						m_userMap.get(userID).setTestReviews(reviews);
					}
				}
			} else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
				review = reviews.get(0);
				synchronized (m_rollbackLock) {
					rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
				}
			}

			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}	
}
