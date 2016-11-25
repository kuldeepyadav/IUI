
ROOTDIR = "/home/rkuldeep/entitydetection/IUI/"
DATABASEPATH = "/home/rkuldeep/entitydetection/IUI/Dataset/"
#DATABASEPATH = "/assignments/IUI/Dataset/"

PICKLEFILEPATH = DATABASEPATH + "Algo_postLabelDict.pickle"

EVALDATASETPATH = ROOTDIR + "evaluationDataset/"


#PATH = "/assignments/stackoverflow/booktext/"
#PATH = "D:\\udacity_2\\tensorflow-udacity-vagrant\\assignments\\stackoverflow\\booktext\\"
wordEmbeddingPath = "/home/rkuldeep/entitydetection/IUI/embeddings/bookEmbeddingModel.bin"
#wordEmbeddingPath = "/assignments/IUI/embeddings/bookEmbeddingModel.bin"

charEmbeddingPath = "/home/kuldeep/entitydetection/IUI/embeddings/"
posNPArrayPath = DATABASEPATH + "PosNPArray.pickle"

maxSeqLen = 99
wordEmbeddingSize = 300
charEmbeddingSize = 40
TRAIN_THRESHOLD = 130000

MODELFOLDERNAME = '/home/rkuldeep/entitydetection/IUI/models/Algo/'

MODELPATH = MODELFOLDERNAME + 'model_specifications.json'
MODELWEIGHTS = MODELFOLDERNAME + 'allBooksKerasModel_49.h5'



