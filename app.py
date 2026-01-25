import streamlit as st
import pandas as pd
import random

from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations
from hybrid_recommendation import hybrid_recommendation_filtering

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Enabled Recommendation Engine",
    page_icon="🤖",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* ========== BACKGROUND & BASE COLORS ========== */
    .main {
        background: #F8F9FA !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #F8F9FA !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #F8F9FA !important;
    }
    
    /* ========== SIDEBAR STYLING ========== */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E8E8E8 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #2D3436 !important;
    }
    
    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        color: #2D3436 !important;
    }
    
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #2D3436 !important;
        font-size: 42px !important;       
        text-align: center;               
        line-height: 1.2;                 
        margin: 20px 0 30px 0;                   
    }

    /* Responsive title adjustments */
    @media (max-width: 1200px) {
        .main-title {
            font-size: 36px !important;
        }
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 28px !important;
        }
    }

    @media (max-width: 480px) {
        .main-title {
            font-size: 22px !important;
        }
    }

    /* Text Colors - ALL WHITE */
    p, span, label, div {
        color: #2D3436 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* ========== PRODUCT CARD STYLING - DARK THEME ========== */
    .product-card {
        background: #FFFFFF !important;
        border-radius: 20px;
        padding: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
        border: 1px solid #F0F0F0;
        position: relative;
    }
    
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.3);
        border-color: #E0E0E0;
    }
    
    .product-tag {
        position: absolute;
        top: 10px;
        left: 10px;
        background: white;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 10px;
        font-weight: 600;
        color: #333;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        z-index: 10;
    }
    
    .product-image-container {
        width: 100%;
        height: 160px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #FAFAFA;
        border-radius: 16px;
        margin-bottom: 12px;
        overflow: hidden;
        padding: 16px;
    }
    
    .product-image {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    
    .product-name {
        font-size: 13px;
        font-weight: 500;
        color: #2D3436 !important;
        margin: 6px 0;
        height: 38px;
        line-height: 1.3;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    .product-brand {
        font-size: 11px;
        color: #7C7C7C !important;
        font-weight: 400;
        margin: 2px 0;
        height: 16px;
    }
    
    .product-rating {
        display: flex;
        align-items: center;
        color: #2D3436 !important;
        font-size: 12px;
        margin: 6px 0;
        height: 18px;
    }
    
    .product-price {
        font-size: 16px;
        font-weight: 700;
        color: #2D3436 !important;
        margin: 4px 0 8px 0;
    }
    
    /* ========== BUTTON STYLING ========== */
    .stButton > button {
        border-radius: 12px !important;
        font-size: 13px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
        border: 1px solid #E0E0E0 !important;
        background: transparent !important;
        color: white !important;
        box-shadow: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        border-color: white !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* ========== SIDEBAR INPUT STYLING ========== */
    [data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #2D3436 !important;
        border: none !important;
        border-radius: 8px !important;
        border-bottom: 2px solid #E0E0E0 !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 400 !important;
    }
    
    [data-testid="stSidebar"] input::placeholder {
        color: #999999 !important;
        opacity: 1 !important;
        font-weight: 400 !important;
    }
    
    [data-testid="stSidebar"] input:focus {
        border: 2px solid #2D3436 !important;
        outline: none !important;
        box-shadow: none !important;
        background-color: #FFFFFF !important;
    }
    
    /* ========== FIX SELECTBOX/DROPDOWN TEXT - WHITE ========== */
    [data-testid="stSelectbox"] label {
        color: #ffffff !important;
    }
    
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #2D3436 !important;
    }
    
    /* Dropdown options */
    [data-baseweb="popover"] {
        background-color: #2D3436 !important;
    }
    
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #2D3436 !important;
    }
    
    [role="option"]:hover {
        background-color: #E0E0E0 !important;
    }
    
    /* ========== FIX EXPANDER/DROPDOWN TEXT ========== */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #2D3436 !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #2D3436 !important;
    }
    
    /* Fix button text in sidebar */
    [data-testid="stSidebar"] button {
        color: #2D3436 !important;
    }
    
    [data-testid="stSidebar"] button p {
        color: #2D3436 !important;
    }
    
    /* ========== STREAMLIT ELEMENTS ========== */
    [data-testid="stMarkdownContainer"] {
        color: #2D3436 !important;
    }
    
    /* ========== ATTRACTIVE USER CARDS ========== */
    .user-cards-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .user-card-item {
        background:#F8F9FA !important;
        border: 1px solid #E0E0E0;
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .user-card-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: white;
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .user-card-item:hover::before {
        transform: scaleX(1);
    }
    
    .user-card-item:hover {
        transform: translateY(-8px);
        border-color: white;
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.3);
    }
    
    .user-avatar-large {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin: 0 auto 16px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .user-id-text {
        font-size: 18px;
        font-weight: 600;
        color: #2D3436 !important;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .user-stats-text {
        font-size: 12px;
        color: #999 !important;
        text-align: center;
    }
    
    /* ========== DROPDOWN MENU STYLING ========== */
    .dropdown-container {
        position: relative;
        display: inline-block;
    }
    
    .dropdown-button {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 12px 20px;
        color: white;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
        transition: all 0.3s ease;
    }
    
    .dropdown-button:hover {
        border-color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* ========== FAVORITES SECTION ========== */
    .favorites-badge {
        background: white;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        margin-left: 8px;
    }

[data-testid="stDialog"] h3 {
                color: #000000 !important;
            }
            [data-testid="stDialog"] p {
                color: #000000 !important;
            }
            [data-testid="stDialog"] strong {
                color: #000000 !important;
            }
            [data-testid="stDialog"] [data-testid="stMarkdownContainer"] {
                color: #000000 !important;
            }
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    raw = pd.read_csv("clean_data.csv")
    return process_data(raw)

df = load_data()

# ================= SESSION STATE =================
if 'show_product_modal' not in st.session_state:
    st.session_state.show_product_modal = False
if 'product_details' not in st.session_state:
    st.session_state.product_details = None
if 'show_user_products' not in st.session_state:
    st.session_state.show_user_products = False
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'show_favorites' not in st.session_state:
    st.session_state.show_favorites = False

# ================= SIDEBAR =================
st.sidebar.header("👤 User Profile")

user_input = st.sidebar.text_input("Enter User ID (0 = Guest)")
user_id = int(user_input) if user_input.isdigit() else 0

top_n = st.sidebar.slider("Number of items", 1, 50, 8, step=4)

# ================= PRODUCT SEARCH IN SIDEBAR =================
st.sidebar.markdown("---")
selected_product = st.sidebar.text_input(
    "Search",
    placeholder="Search products...",
    label_visibility="collapsed"
)

# ================= USER HISTORY (TOP 3) =================
st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Recent History")

if user_id == 0:
    st.sidebar.info("Guest user – no history")
else:
    history = (
        df[df["ID"] == user_id]
        .sort_values("Rating", ascending=False)
        .head(3)
    )

    if history.empty:
        st.sidebar.info("No history available")
    else:
        for _, row in history.iterrows():
            st.sidebar.write(f"• {row['Name']} ⭐ {row['Rating']}")

# ================= DROPDOWN MENU IN SIDEBAR =================
st.sidebar.markdown("---")
st.sidebar.subheader("Your Favourites")

# Create dropdown using expander
if "show_favorites" not in st.session_state:
    st.session_state.show_favorites = False

if "show_user_products" not in st.session_state:
    st.session_state.show_user_products = False

# Favorites button
if st.sidebar.button("❤️ My Favorites", use_container_width=True):
    st.session_state.show_favorites = True
    st.session_state.show_user_products = False
    st.rerun()

# ================= MAIN =================
st.markdown('<h1 class="main-title">🤖 AI Enabled Recommendation Engine For An E-Commerce Platform</h1>', unsafe_allow_html=True)

# ================= SORTING FUNCTION =================
def sort_products(data, sort_option):
    if data.empty:
        return data
    
    data = data.copy()
    
    if sort_option == "Recommended":
        return data
    elif sort_option == "Popularity":
        if "ReviewCount" in data.columns:
            return data.sort_values("ReviewCount", ascending=False)
        return data
    elif sort_option == "Price: High to Low":
        data['_sort_price'] = data['Name'].apply(lambda x: (hash(x) % 300) / 100 + 2.5)
        sorted_data = data.sort_values('_sort_price', ascending=False).drop('_sort_price', axis=1)
        return sorted_data
    elif sort_option == "Price: Low to High":
        data['_sort_price'] = data['Name'].apply(lambda x: (hash(x) % 300) / 100 + 2.5)
        sorted_data = data.sort_values('_sort_price', ascending=True).drop('_sort_price', axis=1)
        return sorted_data
    elif sort_option == "Customer Rating":
        if "Rating" in data.columns:
            return data.sort_values("Rating", ascending=False)
        return data
    return data

# ================= DISPLAY FUNCTION =================
def display_products(title, data, section_id, show_sort=True):
    if data.empty:
        st.info("No products found")
        return

    st.subheader(title)
    
    # Add sort dropdown above products
    if show_sort:
        col1, col2 = st.columns([3, 1])
        with col2:
            sort_options = [
                "Recommended",
                "What's New",
                "Popularity",
                "Better Discount",
                "Price: High to Low",
                "Price: Low to High",
                "Customer Rating"
            ]
            
            sort_key = f"sort_{section_id}"
            selected_sort = st.selectbox(
                "Sort by:",
                sort_options,
                key=sort_key,
                label_visibility="visible"
            )
            
            # Apply sorting
            data = sort_products(data, selected_sort)
    
    # Create rows of 4 columns
    for row_start in range(0, len(data), 4):
        cols = st.columns(4)
        row_data = data.iloc[row_start:row_start + 4]
        
        for col_idx, (idx, row) in enumerate(row_data.iterrows()):
            with cols[col_idx]:
                # Product Card Container
                img_url = row.get('ImageURL', 'https://via.placeholder.com/150x150')
                if not isinstance(img_url, str) or not img_url.startswith('http'):
                    img_url = 'https://via.placeholder.com/150x150'
                
                # Truncate product name if too long
                product_name = row['Name']
                display_name = product_name
                if len(product_name) > 45:
                    display_name = product_name[:42] + "..."
                
                # Generate consistent price for each product
                price_value = (hash(row['Name']) % 300) / 100 + 2.5
                price = f"${price_value:.1f}"
                
                # Unique button keys
                view_key = f"view_{section_id}_{idx}_{row_start}_{col_idx}"
                fav_key = f"fav_{section_id}_{idx}_{row_start}_{col_idx}"
                
                # Check if in favorites
                is_favorite = product_name in st.session_state.favorites
                
                # Create card HTML
                card_html = f"""
                <div class="product-card">
                    <div class="product-tag">50% off</div>
                    <div class="product-image-container">
                        <img src="{img_url}" 
                             class="product-image" 
                             onerror="this.src='https://via.placeholder.com/150x150'">
                    </div>
                    <div class="product-name">{display_name}</div>
                    <div class="product-brand">{row.get('Brand', 'Brand')}</div>
                    <div class="product-rating">
                        <span>⭐ {row.get('Rating', 'N/A')}</span>
                    </div>
                    <div class="product-price">{price}</div>
                </div>
                """

                st.markdown(card_html, unsafe_allow_html=True)
                
                # Action buttons OUTSIDE the card - SYMBOLS ONLY
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("📋", key=view_key, use_container_width=True):
                        st.session_state.show_product_modal = True
                        st.session_state.product_details = {
                            'Name': product_name,
                            'Brand': row.get('Brand', 'Unknown'),
                            'Rating': row.get('Rating', 'N/A'),
                            'ReviewCount': row.get('ReviewCount', 0),
                            'ImageURL': img_url,
                            'Price': price
                        }
                        st.rerun()
                
                with btn_col2:
                    fav_icon = "💖" if is_favorite else "🤍"
                    if st.button(fav_icon, key=fav_key, use_container_width=True):
                        if is_favorite:
                            st.session_state.favorites.remove(product_name)
                            st.toast("💔 Removed from favorites")
                        else:
                            st.session_state.favorites.append(product_name)
                            st.toast("💖 Added to favorites!")
                        st.rerun()

# ================= PRODUCT DETAILS MODAL =================
if st.session_state.show_product_modal and st.session_state.product_details:
    product = st.session_state.product_details
    
    @st.dialog("📦 Product Details")
    def show_product_modal():
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(product['ImageURL'], use_container_width=True)
        
        with col2:
            st.markdown(f"### {product['Name']}")
            st.write("")
            st.markdown(f"**🏷️ Brand:** {product['Brand']}")
            st.markdown(f"**⭐ Rating:** {product['Rating']} ({product['ReviewCount']} reviews)")
            st.markdown(f"**💰 Price:** {product['Price']}")
        
        if st.button("Close", use_container_width=True):
            st.session_state.show_product_modal = False
            st.session_state.product_details = None
            st.rerun()
    
    show_product_modal()

# ================= SHOW FAVORITES =================
if st.session_state.show_favorites:
    st.markdown("---")
    
    if st.button("← Back to Home", use_container_width=False):
        st.session_state.show_favorites = False
        st.rerun()
    
    st.markdown("### 💖 My Favorites")
    st.markdown("---")
    
    if st.session_state.favorites:
        # Filter products that are in favorites
        favorite_products = df[df['Name'].isin(st.session_state.favorites)].drop_duplicates(subset=['Name'])
        display_products("", favorite_products, "favorites", show_sort=False)
    else:
        st.info("You haven't added any favorites yet!")

# ================= USER TOP RATED / TRENDING =================
elif not st.session_state.show_user_products:
    if user_id == 0:
        top_items = get_top_rated_items(df, top_n)
        display_products("🔥 Trending Products", top_items, "trending", show_sort=True)
    else:
        user_top = (
            df[df["ID"] == user_id]
            .sort_values("Rating", ascending=False)
            .head(top_n)
        )
        display_products("⭐ Your Top Rated Products", user_top, "user_top", show_sort=True)

    # ================= SIMILAR PRODUCTS =================
    if selected_product and selected_product.strip():
        st.markdown("---")
        matching_products = df[df['Name'].str.contains(selected_product, case=False, na=False)]
        
        if not matching_products.empty:
            exact_product_name = matching_products.iloc[0]['Name']
            similar_products = content_based_recommendation(
                df,
                item_name=exact_product_name,
                top_n=top_n
            )
            display_products(f"🧩 Similar Products to '{selected_product}'", similar_products, "similar", show_sort=True)
        else:
            st.info(f"No products found matching '{selected_product}'. Try searching for another product.")

# ================= HELPER FUNCTION =================
def get_recommended_products_from_user(df, target_user, similar_user):
    """Get products that similar user liked but target user hasn't interacted with"""
    similar_user_likes = df[
        (df["ID"] == similar_user) & (df["Rating"] >= 4)
    ][["ProdID", "Name", "Brand", "Rating", "ReviewCount", "ImageURL"]]

    target_user_products = df[df["ID"] == target_user]["ProdID"].unique()

    recommendations = similar_user_likes[
        ~similar_user_likes["ProdID"].isin(target_user_products)
    ]

    if recommendations.empty:
        return pd.DataFrame()

    return recommendations.head(4)

# ================= SHOW USER PRODUCTS =================
if st.session_state.show_user_products and st.session_state.selected_user:
    sim_user = st.session_state.selected_user
    
    st.markdown("---")
    
    if st.button("← Back", use_container_width=False):
        st.session_state.show_user_products = False
        st.session_state.selected_user = None
        st.rerun()
    
    st.markdown(f"### 👤 User {sim_user}'s Recommendations")
    st.markdown("---")
    
    recommended_products = get_recommended_products_from_user(df, user_id, sim_user)
    
    if not recommended_products.empty:
        display_products(
            "", 
            recommended_products, 
            f"user_{sim_user}",
            show_sort=False
        )
    else:
        st.info(f"User {sim_user} hasn't rated any products you haven't tried yet.")

# ================= ATTRACTIVE USERS WITH SIMILAR TASTE =================
if user_id != 0 and not st.session_state.show_user_products and not st.session_state.show_favorites:
    st.markdown("---")
    st.markdown(
        "<h2 style='font-family: Poppins; font-size: 28px; margin-bottom: 20px;'>👥 Users with Similar Taste</h2>",
        unsafe_allow_html=True
    )

    similar_users = (
        df[df["ProdID"].isin(df[df["ID"] == user_id]["ProdID"])]
        .groupby("ID")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    similar_users = [u for u in similar_users if u != user_id][:12]

    if similar_users:
        # Display attractive user cards
        cols_per_row = 4
        for i in range(0, len(similar_users), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, sim_user in enumerate(similar_users[i:i+cols_per_row]):
                with cols[j]:
                    # Get stats
                    common_items = df[
                        (df["ID"] == sim_user) &
                        (df["ProdID"].isin(df[df["ID"] == user_id]["ProdID"]))
                    ].shape[0]
                    
                    recommended_products = get_recommended_products_from_user(df, user_id, sim_user)
                    product_count = len(recommended_products)
                    
                    # Create attractive user card
                    user_card_html = f"""
                    <div class="user-card-item">
                        <div class="user-avatar-large">👤</div>
                        <div class="user-id-text">User {sim_user}</div>
                        <div class="user-stats-text">{common_items} common items</div>
                        <div class="user-stats-text">{product_count} recommendations</div>
                    </div>
                    """
                    
                    st.markdown(user_card_html, unsafe_allow_html=True)
                    
                    if st.button(
                        "View Products →",
                        key=f"user_btn_{sim_user}",
                        use_container_width=True
                    ):
                        st.session_state.show_user_products = True
                        st.session_state.selected_user = sim_user
                        st.rerun()
    else:
        st.info("No similar users found")
