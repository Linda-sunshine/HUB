package structures;

public class DPParameter {

    public String m_prefix = "./data/CoLinAdapt";
    public String m_data = "YelpNew";
    public String m_model = "hub";

    public double m_adaptRatio = 0.5;
    public int m_nuOfIterations = 30;
    public int m_M = 6;
    public int m_burnin = 10;
    public int m_thinning = 3;
    public double m_sdA = 0.0425;
    public double m_sdB = 0.0425;

    // Concentration parameter
    public double m_alpha = 0.01;
    public double m_eta = 0.05;
    public double m_beta = 0.01;

    public double m_eta1 = 0.05;
    public double m_eta2 = 0.05;
    public double m_eta3 = 0.05;
    public double m_eta4 = 0.05;

    // MTCLRWithDP, MTCLRWithHDP
    public double m_q = 0.1; // global parameter.

    // paramters for feature groups
    public int m_fv = 800;
    public int m_fvSup = 5000;
    public int m_lmTopK = 1000;
    public String m_fs = "DF";

    // used in mmb model, sparsity parameter
    public double m_rho = 0.05;
    public boolean m_jointAll = false;

    public DPParameter(String argv[]) {

        int i;

        //parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-')
                break;
            else if (++i >= argv.length)
                exit_with_help();
            else if (argv[i - 1].equals("-data"))
                m_data = argv[i];
            else if (argv[i - 1].equals("-eta1"))
                m_eta1 = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-eta2"))
                m_eta2 = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-eta3"))
                m_eta3 = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-eta4"))
                m_eta4 = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-model"))
                m_model = argv[i];

            else if (argv[i - 1].equals("-fv"))
                m_fv = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-fvSup"))
                m_fvSup = Integer.valueOf(argv[i]);

            else if (argv[i - 1].equals("-M"))
                m_M = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-alpha"))
                m_alpha = Double.valueOf(argv[i]);

            else if (argv[i - 1].equals("-nuI"))
                m_nuOfIterations = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-sdA"))
                m_sdA = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-sdB"))
                m_sdB = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-burn"))
                m_burnin = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-thin"))
                m_thinning = Integer.valueOf(argv[i]);
            else if (argv[i - 1].equals("-eta"))
                m_eta = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-beta"))
                m_beta = Double.valueOf(argv[i]);
            else if (argv[i - 1].equals("-rho"))
                m_rho = Double.parseDouble(argv[i]);
            else
                exit_with_help();
        }
    }

    private void exit_with_help() {
        System.out.print("Usage: java execution [options] training_folder\n"
                + "options:\n"
                + "-data: specific the dataset used for training (default YelpNew)\noption: Amazon, YelpNew\n"
                + "-eta1: coefficient for the scaling in each user group's regularization (default 0.05)\n"
                + "-eta2: coefficient for the shifting in each user group's regularization (default 0.05)\n"
                + "-eta3: coefficient for the scaling in super user's regularization (default 0.05)\n"
                + "-eta4: coefficient for the shifting in super user's regularization (default 0.05)\n"
                + "-model: specific training model,\noption: Base-base, MT-SVM-mtsvm, GBSSL-gbssl, HUB-hub"
                + "MTLinAdapt+kMeans-mtclinkmeans,  cLinAdapt-mtclindp, cLinAdapt+HDP-mtclinhdp (default hub)\n"
                + "-M: the size of the auxiliary variables in the posterior inference of the group indicator (default 6)\n"
                + "-alpha: concentraction parameter for the first-layer DP, i.e., collective identities (default 0.01)\n"
                + "-beta: concentraction parameter for the prior Dirichlet Distribution of language model (default 0.05)\n"
                + "-eta: concentraction parameter for the second-layer DP, i.e., user mixture (default 0.05)\n"
                + "-nuI: number of iterations for sampling (default 30)\n"
                + "-sdA: variance for the normal distribution for the prior of shifting parameters (default 0.0425)\n"
                + "-sdB: variance for the normal distribution for the prior of scaling parameters (default 0.0425)\n"
                + "-burn: number of iterations in burn-in period (default 5)\n"
                + "-thin: thinning of sampling chain (default 5)\n"
                + "-rho: the sparsity parameter for block model (default 0.05)\n"
                + "--------------------------------------------------------------------------------\n"
        );
        System.exit(1);
    }
}
