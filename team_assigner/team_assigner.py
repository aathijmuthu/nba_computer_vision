from sklearn.cluster import KMeans

class TeamAssigner:

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters = 5, init = "k-means++", n_init = 1).fit(image_2d)
        kmeans.fit(image_2d)

        return kmeans

    def get_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[:int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(top_half)
        labels = kmeans.labels_
        segmented_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        player_color = kmeans.cluster_centers_[4]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player in player_detections.items():
            bbox = player["bbox"]
            player_color = self.get_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters = 2, init = "k-means++", n_init = 1).fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id

        return team_id
