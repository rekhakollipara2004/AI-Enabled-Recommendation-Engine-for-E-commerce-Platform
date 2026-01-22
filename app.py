import streamlit as st
import pandas as pd

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
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');
    
    .main {
        background-color: #000000 !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    
    /* Sidebar Black Theme */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Global Font Styles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Comic Neue', 'Comic Sans MS', cursive, sans-serif !important;
        color: #ffffff !important;
    }
    
    .main-title {
    font-family: 'sans-serif';
    font-weight: 700;
    color: #ffffff !important;
    font-size: 48px !important;       
    text-align: center;               
    line-height: 1.2;                 
    margin: 20px 0;                   
}

/* Responsive adjustments */
@media (max-width: 1200px) {
    .main-title {
        font-size: 40px !important;
    }
}

@media (max-width: 768px) {
    .main-title {
        font-size: 32px !important;
    }
}

@media (max-width: 480px) {
    .main-title {
        font-size: 24px !important;
    }
}

    /* Text Colors */
    p, span, label, div {
        color: #ffffff !important;
    }
    
    /* Product Card Styling - Black Theme */
    .product-card {
        background: #1a1a1a !important;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(255,255,255,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        margin-bottom: 16px;
        border: 1px solid #333;
    }
    
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 16px rgba(255,106,53,0.3);
        border-color: white;
    }
    
    .product-image-container {
        width: 100%;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #2a2a2a;
        border-radius: 12px;
        margin-bottom: 10px;
        overflow: hidden;
        padding: 8px;
    }
    
    .product-image {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    
    .product-name {
        font-size: 13px;
        font-weight: 600;
        color: #ffffff !important;
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
        font-size: 12px;
        color: white !important;
        font-weight: 500;
        margin: 3px 0;
        height: 18px;
    }
    
    .product-rating {
        display: flex;
        align-items: center;
        gap: 4px;
        color: #FFA500 !important;
        font-size: 12px;
        margin: 6px 0;
        height: 18px;
    }
    
    /* View Details Button - Larger Size */
    button[kind="secondary"] {
        background-color: black !important;
        color: white !important;
        border: 1px !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        min-height: 45px !important;
    }
    
    button[kind="secondary"]:hover {
        background-color: black !important;
        color: white !important;
        transform: translateY(-1px);
    }
    
    /* Modal Dialog - Smaller Size */
    [data-testid="stDialog"] {
        max-width: 350px !important;
    }
    
    [data-testid="stDialog"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        padding: 20px !important;
        border: 1px solid #ddd !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stDialog"] h2 {
        color: #000000 !important;
        font-size: 18px !important;
    }
    
    [data-testid="stDialog"] p,
    [data-testid="stDialog"] div,
    [data-testid="stDialog"] span {
        color: #000000 !important;
    }
    
    /* Similar Users Card Styling */
    .user-card {
    background: black;
    border-radius: 16px;
    padding: 10px;
    color: white;
    text-align: center;
    border: 1px solid white;
    height: 260px;   
    width: 200px;              
    display: flex;
    flex-direction: column;
    }
    
    .user-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(255,255,255,0.1);
    }
    
    .user-avatar {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px;
        font-size: 28px;
    }
    
    .user-id {
        font-size: 18px;
        font-weight: 700;
        margin: 8px 0;
        font-family: 'Comic Neue', 'Comic Sans MS', cursive;
        color: #ffffff !important;
    }
    
    .user-label {
        font-size: 12px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ffffff !important;
    }
    
    /* Custom Search Input Styling */
    [data-testid="stSidebar"] input {
        background-color: #1a1a1a !important;
        color: #cccccc !important;
        border: none !important;
        border-bottom: 2px solid #333333 !important;
        border-radius: 0px !important;
        padding: 8px 0px !important;
        font-size: 16px !important;
        font-family: 'Comic Neue', 'Comic Sans MS', cursive, sans-serif !important;
        font-weight: 400 !important;
    }
    
    [data-testid="stSidebar"] input::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
        font-weight: 400 !important;
    }
    
    [data-testid="stSidebar"] input:focus {
        border-bottom: 2px solid #555555 !important;
        outline: none !important;
        box-shadow: none !important;
        background-color: #1a1a1a !important;
    }
    
    /* Input Fields */
    input, select, textarea {
        background-color: #1a1a1a !important;
        color: #cccccc !important;
        border: none !important;
        border-bottom: 2px solid #333333 !important;
        border-radius: 0px !important;
    }
    
    /* Streamlit Elements */
    [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
            
    .user-card [data-testid="stMarkdownContainer"] {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    raw = pd.read_csv("clean_data.csv")
    return process_data(raw)

df = load_data()

# ================= SESSION STATE FOR MODAL =================
if 'show_modal' not in st.session_state:
    st.session_state.show_modal = False
    st.session_state.selected_product_data = None
if 'modal_counter' not in st.session_state:
    st.session_state.modal_counter = 0

# ================= SIDEBAR =================
st.sidebar.header("👤 User Profile")

user_input = st.sidebar.text_input("Enter User ID (0 = Guest)")
user_id = int(user_input) if user_input.isdigit() else 0

top_n = st.sidebar.slider("Number of items", 1, 50, 2, step=3)

# ================= PRODUCT SEARCH IN SIDEBAR =================
st.sidebar.markdown("---")
selected_product = st.sidebar.text_input(
    "Search",
    placeholder="Search",
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

# ================= MAIN =================
st.markdown('<h1 class="main-title">🤖 AI Enabled Recommendation Engine for E-commerce Platform</h1>', unsafe_allow_html=True)

# ================= DISPLAY FUNCTION WITH MODAL =================
def display_products(title, data, section_id):
    if data.empty:
        st.info("No products found")
        return

    st.subheader(title)
    
    # Create rows of 4 columns
    for row_start in range(0, len(data), 4):
        cols = st.columns(4)
        row_data = data.iloc[row_start:row_start + 4]
        
        for col_idx, (idx, row) in enumerate(row_data.iterrows()):
            with cols[col_idx]:
                # Product Card Container
                img_url = row.get('ImageURL', 'https://via.placeholder.com/100x200')
                if not isinstance(img_url, str) or not img_url.startswith('http'):
                    img_url = 'https://via.placeholder.com/300x200'
                
                # Truncate product name if too long
                product_name = row['Name']
                if len(product_name) > 50:
                    product_name = product_name[:47] + "..."
                
                card_html = f"""
                <div class="product-card">
                    <div class="product-image-container">
                        <img src="{img_url}" 
                             class="product-image" 
                             onerror="this.src='https://via.placeholder.com/300x200'">
                    </div>
                    <div class="product-name">{product_name}</div>
                    <div class="product-brand">{row.get('Brand', 'Unknown Brand')}</div>
                    <div class="product-rating">
                        ⭐ {row.get('Rating', 'N/A')} <span style="color: #999;">({row.get('ReviewCount', 0)})</span>
                    </div>
                </div>
                """

                st.markdown(card_html, unsafe_allow_html=True)

                # View Button with unique key using index
                unique_key = f"view_{section_id}_{idx}_{row_start}_{col_idx}"
                if st.button("View Details", key=unique_key, use_container_width=True):
                    st.session_state.show_modal = True
                    st.session_state.selected_product_data = row
                    st.session_state.modal_counter += 1
                    st.rerun()
 
# ================= MODAL DIALOG (SMALLER SIZE) =================
@st.dialog("📦 Product Details")
def show_product_modal(product_data):
    # Product Image (smaller)
    img = product_data.get("ImageURL")
    if isinstance(img, str) and img.startswith("http"):
        st.image(img, width=150)
    else:
        st.image("https://via.placeholder.com/300x200", width=150)
    
    # Product Name (FULL NAME - NO TRUNCATION)
    product_name = product_data['Name']
    st.markdown(f"**{product_name}**")
    
    # Brand and Rating in compact layout
    st.markdown(f"🏷️ {product_data.get('Brand', 'Unknown')}")
    st.markdown(f"⭐ {product_data.get('Rating', 'N/A')} ({product_data.get('ReviewCount', 0)} reviews)")
    
    st.markdown("---")
    

# Show modal if triggered
if st.session_state.show_modal and st.session_state.selected_product_data is not None:
    show_product_modal(st.session_state.selected_product_data)
    st.session_state.show_modal = False

# ================= USER TOP RATED / TRENDING =================
if user_id == 0:
    top_items = get_top_rated_items(df, top_n)
    display_products("🔥 Trending Products", top_items, "trending")
else:
    user_top = (
        df[df["ID"] == user_id]
        .sort_values("Rating", ascending=False)
        .head(top_n)
    )
    display_products("⭐ Your Top Rated Products", user_top, "user_top")

# ================= SIMILAR PRODUCTS (CONTENT ONLY) =================
if selected_product and selected_product.strip():
    st.markdown("---")
    # Try to find matching product (case-insensitive partial match)
    matching_products = df[df['Name'].str.contains(selected_product, case=False, na=False)]
    
    if not matching_products.empty:
        # Use the first matching product name
        exact_product_name = matching_products.iloc[0]['Name']
        similar_products = content_based_recommendation(
            df,
            item_name=exact_product_name,
            top_n=top_n
        )
        display_products(f"🧩 Similar Products to '{selected_product}'", similar_products, "similar")
    else:
        st.info(f"No products found matching '{selected_product}'. Try copying a product name from the list.")

# ================= SIMILAR PRODUCTS =================
def get_recommended_products_from_user(df, target_user, similar_user):
    similar_user_likes = df[
        (df["ID"] == similar_user) & (df["Rating"] >= 4)
    ][["ProdID", "Name"]]

    target_user_products = df[df["ID"] == target_user]["ProdID"].unique()

    recommendations = similar_user_likes[
        ~similar_user_likes["ProdID"].isin(target_user_products)
    ]

    if recommendations.empty:
        return None

    return recommendations.iloc[0]["Name"]


# ================= SIMILAR USERS =================
if user_id != 0:
    st.markdown("---")
    st.markdown(
        "<h2 style='font-family: Comic Neue;'>👥 Users with Similar Taste</h2>",
        unsafe_allow_html=True
    )

    similar_users = (
        df[df["ProdID"].isin(df[df["ID"] == user_id]["ProdID"])]
        .groupby("ID")
        .size()
        .sort_values(ascending=False)
        .head(10)  # exactly 6 users
        .index.tolist()
    )

    similar_users = [u for u in similar_users if u != user_id][:8]

    if similar_users:
        cols = st.columns(4)
        user_emojis = ["👤", "🧑🏻‍💼", "🤵🏻", "🤵🏻‍♀️", "👩🏼‍💼", "👤"]

        for idx, sim_user in enumerate(similar_users):
            with cols[idx % 4]:
                emoji = user_emojis[idx % len(user_emojis)]

                # COMMON ITEMS COUNT
                common_items = df[
                    (df["ID"] == sim_user) &
                    (df["ProdID"].isin(df[df["ID"] == user_id]["ProdID"]))
                ].shape[0]

                # GET ONE RECOMMENDED PRODUCT
                product_name = get_recommended_products_from_user(
                    df, user_id, sim_user
                )

                # Simple card with USER ID + Product name
                st.markdown(f"""
                <div class="user-card">
                    <div class="user-avatar">{emoji}</div>
                    <div class="user-label">USER ID</div>
                    <div class="user-id">{sim_user}</div>
                    <p style="font-size:12px;margin-top:4px;">
                        🎯 {common_items} common items
                    </p>
                    <p style="margin-top:6px;font-size:13px;color:#ffffff;font-weight:500;">
                        {product_name if product_name else "No new recommendation"}
                    </p><br>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No similar users found")
