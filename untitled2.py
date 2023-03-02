import pickle
pickle_out = open("modeltest.pkl", "wb")
pickle.dump(gbc, pickle_out)
pickle_out.close()