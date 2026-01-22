# collaborative_based_filtering.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering_recommendations(data, target_user_id, top_n=10):

    if target_user_id not in data["ID"].values:
        return pd.DataFrame()

    # User-Item Matrix
    user_item_matrix = data.pivot_table(
        index='ID',
        columns='ProductIndex',
        values='Rating',
        aggfunc='mean',
        fill_value=0
    )

    # Similarity
    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    similarity_scores = user_similarity[target_user_index]
    similar_users = similarity_scores.argsort()[::-1][1:]

    recommended_products = set()

    for user_idx in similar_users:
        rated = user_item_matrix.iloc[user_idx]
        unrated = user_item_matrix.iloc[target_user_index] == 0
        products = rated[(rated > 0) & unrated].index.tolist()
        recommended_products.update(products)
        if len(recommended_products) >= top_n:
            break

    # Map back to real products
    recommendations = (
        data[data['ProductIndex'].isin(recommended_products)]
        .drop_duplicates('ProductIndex')
        [['Name', 'Brand', 'Rating', 'ReviewCount', 'ImageURL', 'Description']]
        .head(top_n)
    )

    return recommendations
if __name__ == "__main__":
    import pandas as pd
    from preprocess_data import process_data
    from collaborative_based_filtering import collaborative_filtering_recommendations

    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)

    target_user_id = 15
    top_n = 5

    recommendations = collaborative_filtering_recommendations(
        data, target_user_id, top_n
    )

    print(recommendations[['Name', 'Brand', 'Rating']])
    print("\n✅ Collaborative Filtering is working")
