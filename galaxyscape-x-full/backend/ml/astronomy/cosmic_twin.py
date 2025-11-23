"""Cosmic Twin finder."""
import numpy as np

def map_human_features_to_embedding(human_vector, encoder):
    """Map human features to embedding space."""
    if hasattr(encoder, 'encoder'):
        # PyTorch model
        try:
            import torch
            tensor = torch.FloatTensor([human_vector])
            with torch.no_grad():
                embedding = encoder.encoder(tensor)
            return embedding.numpy()[0]
        except:
            pass
    
    # Fallback: return normalized vector
    vec = np.array(human_vector)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity."""
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))

def find_top_k_twins(human_embedding, star_embeddings, k=5):
    """Find top k matching stars."""
    scores = []
    for star_id, star_embedding in star_embeddings.items():
        similarity = cosine_similarity(human_embedding, star_embedding)
        scores.append((star_id, similarity))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def format_twin_payload(twins):
    """Format results for API."""
    star_names = {
        'star_001': 'Alpha Centauri',
        'star_002': 'Sirius',
        'star_003': 'Vega',
        'star_004': 'Betelgeuse',
        'star_005': 'Rigel'
    }
    
    return [
        {
            'star_id': star_id,
            'similarity': float(score),
            'name': star_names.get(star_id, f'Star {star_id}')
        }
        for star_id, score in twins
    ]

