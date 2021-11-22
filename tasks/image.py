class Image:
    def __init__(self, filename, image_arr = None, type = None, subject_id = None, image_id = None, features = None):
        self.filename = filename
        self.type = type
        self.subject_id = subject_id
        self.image_id = image_id
        self.image_arr = image_arr
        self.features = features
        self.latent_features = None

    def set_features(self, features):
        self.features = features

    def set_latent_features(self, latent_features):
        self.latent_features = latent_features

