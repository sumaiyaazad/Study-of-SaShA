from detections.Number_Of_Predicton_Differences_Detector import PredictionDifferenceDetector
from utils.data_loader import *

data, users, items = load_data_yahoo_movies()
predictor = PredictionDifferenceDetector(data)
npd_filename = "experiment/experiment_yahoo_movies_npd.txt"
fake_profile_filename = "experiment/experiment_yahoo_movies_fake_profile.txt"
fake_profile_list = "experiment/experiment_yahoo_movies_fake_profile_list.txt"
predictor.predict_fake_profiles(npd_filename, fake_profile_filename, fake_profile_list)
