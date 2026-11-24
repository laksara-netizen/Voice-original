
import pickle
import numpy as np
import os
import logging
from datetime import datetime
from librosa.sequence import dtw
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class VoiceMatcher:
    def __init__(self):
        self.alpha = 40.0
        self.threshold = 0.3
        self.weights = (0.5, 0.5)

    def dtw_distance(self, A, B):
        try:
            if A.ndim != 2 or B.ndim != 2:
                logger.warning(f"DTW input must be 2D arrays. Got A={A.shape}, B={B.shape}")
                return float("inf")
            #if A.shape[1] != B.shape[1]:
                #logger.warning(f"DTW feature dimension mismatch: A={A.shape}, B={B.shape}")
                #return float("inf")
            #if A.shape[0] < A.shape[1]: A = A.T
            #if B.shape[0] < B.shape[1]: B = B.T
            logger.info(f"DTW input shapes: test={A.shape}, enrolled={B.shape}")
            D, _ = dtw(A, B, metric='euclidean')
            dist = D[-1, -1]
            logger.info(f"DTW raw distance: {dist:.2f}")
            return dist
        except Exception as e:
            logger.error(f"DTW error: {str(e)}")
            return float("inf")

    def save_voice_sample(self, user_id, sample_number, features_dict):
        try:
            samples_dir = f"voiceprints/user_{user_id}_samples"
            os.makedirs(samples_dir, exist_ok=True)
            path = os.path.join(samples_dir, f"sample_{sample_number}.pkl")
            data = {
                'matrix': features_dict['matrix'],
                'stats': features_dict['stats'],
                'timestamp': datetime.utcnow()
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved voice sample {sample_number} for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving voice sample: {str(e)}")
            return False

    def get_sample_count(self, user_id):
        path = f"voiceprints/user_{user_id}_samples"
        return len([f for f in os.listdir(path) if f.endswith('.pkl')]) if os.path.exists(path) else 0

    def create_voiceprint(self, user_id):
        try:
            path = f"voiceprints/user_{user_id}_samples"
            if not os.path.exists(path):
                logger.error("Samples path not found.")
                return False
            matrices, stats = [], []
            for file in os.listdir(path):
                if file.endswith('.pkl'):
                    with open(os.path.join(path, file), 'rb') as f:
                        data = pickle.load(f)
                        matrices.append(data['matrix'])
                        stats.append(data['stats'])
            if len(matrices) < 3:
                logger.warning(f"Not enough samples to create voiceprint for user {user_id}")
                return False
            voiceprint = {
                'matrix_samples': matrices,
                'stat_mean': np.mean(stats, axis=0)
            }
            with open(f"voiceprints/user_{user_id}_voiceprint.pkl", 'wb') as f:
                pickle.dump(voiceprint, f)
            logger.info(f"Created hybrid voiceprint for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating voiceprint: {str(e)}")
            return False

    def authenticate_voice(self, user_id, test_features):
        try:
            path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if not os.path.exists(path):
                return False, 0.0
            with open(path, 'rb') as f:
                vp = pickle.load(f)

            dtw_dists = [self.dtw_distance(test_features['matrix'], m) for m in vp['matrix_samples']]
            dtw_valid = [d for d in dtw_dists if np.isfinite(d)]
            dtw_avg = np.mean(dtw_valid) if dtw_valid else float("inf")
            dtw_score = np.exp(-dtw_avg / self.alpha) if np.isfinite(dtw_avg) else 0.0

            stat_score = cosine_similarity(
                test_features['stats'].reshape(1, -1),
                vp['stat_mean'].reshape(1, -1)
            )[0][0]

            final_score = self.weights[0] * dtw_score + self.weights[1] * stat_score
            is_match = final_score >= self.threshold

            logger.info(f"Hybrid score: DTW={dtw_score:.3f}, Stats={stat_score:.3f}, Final={final_score:.3f}, Match={is_match}")
            return is_match, final_score
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False, 0.0

    def clear_user_voiceprint(self, user_id):
        try:
            vp_path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if os.path.exists(vp_path): os.remove(vp_path)
            samples_dir = f"voiceprints/user_{user_id}_samples"
            if os.path.exists(samples_dir):
                for f in os.listdir(samples_dir):
                    os.remove(os.path.join(samples_dir, f))
                os.rmdir(samples_dir)
            logger.info(f"Cleared voiceprint for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing voiceprint: {str(e)}")
            return False

    def get_voiceprint_info(self, user_id):
        try:
            path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if not os.path.exists(path): return None
            with open(path, 'rb') as f:
                vp = pickle.load(f)
            return {
                'sample_count': len(vp['matrix_samples']),
                'stat_shape': vp['stat_mean'].shape
            }
        except Exception as e:
            logger.error(f"Error getting voiceprint info: {str(e)}")
            return None
