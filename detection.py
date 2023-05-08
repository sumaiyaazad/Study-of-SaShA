from detections.Number_Of_Predicton_Differences_Detector import PredictionDifferenceDetector
from detections.PCA_Detector import PCAShillingAttackDetector
from utils.data_loader import *

data, users, items = load_data_yahoo_movies()
fake_profile_file = "experiment/experiment_yahoo_movies_fake_profile_list"

npd_predictor = PredictionDifferenceDetector(data, 5)
fake_profile_list = npd_predictor.predict_fake_profiles(fake_profile_file+"_npd.txt")
print("-------------------NPD Top Fake Profiles------------------------------------------")
print(fake_profile_list)

pca_predictor = PCAShillingAttackDetector(data, 10)
print("------------------PCA Top Fake Profiles--------------------------------------------")
fake_profile_list = pca_predictor.predict_fake_profiles(fake_profile_file+"_pca.txt")
print(fake_profile_list)


